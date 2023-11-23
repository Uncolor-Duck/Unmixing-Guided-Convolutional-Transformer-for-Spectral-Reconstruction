import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.acti = F.relu

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = self.acti(out)
        return out


class RAM(nn.Module):
    def __init__(self, in_dim, out_dim, group):
        super(RAM, self).__init__()
        self.out_dim = out_dim

        #self.inConv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, dilation=2, padding=2, bias=False)
        self.inConv_cn = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3,
                                dilation=2, padding=2, bias=False)
        self.inConv_at = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                   bias=False, groups=group)

        self.mlp = nn.Sequential(nn.Linear(out_dim, out_dim * 2, bias=False),
                                 nn.Linear(out_dim * 2, out_dim * 2, bias=False),
                                 nn.Linear(out_dim * 2, out_dim, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x_t, x_c):
        B, _, H, W = x_t.shape
        avgpool = nn.AvgPool1d(H * W)
        x_t = self.inConv_at(x_t)
        x_att = torch.reshape(x_t, [B, -1, H * W])
        x_att = torch.squeeze(avgpool(x_att), -1)
        x_att = self.mlp(x_att)
        x_att = torch.reshape(self.sig(x_att), [B, -1, 1, 1])
        x_c = self.inConv_cn(x_c)
        x_out = x_t + torch.mul(x_att, x_c)
        return x_out


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x_in = x_in.permute(0, 2, 3, 1)
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out = out_c.permute(0, 3, 1, 2)
        return out

class AbunBlock(nn.Module):
    def __init__(self, in_dim, out_dim, group):
        super(AbunBlock, self).__init__()
        dim = in_dim
        self.attn = MS_MSA(dim=in_dim, dim_head=dim, heads=1)
        self.resn = ResBlock(inchannel=in_dim, outchannel=out_dim)
        self.RAM = RAM(in_dim=in_dim, out_dim=out_dim, group=group)
        self.FFN = FeedForward(dim=in_dim)

    def forward(self, x):
        x_t = self.attn(x) + x
        x_t = self.FFN(x_t) + x_t
        x_c = self.resn(x)
        x_out = self.RAM(x_t, x_c)
        return x_out

class Swin(nn.Module):
    def __init__(self, in_dim, out_dim, stage, dim=32):
        super(Swin, self).__init__()
        stage_dim = dim
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=stage_dim, kernel_size=1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])

        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                AbunBlock(in_dim=dim, out_dim=dim * 2, group=stage_dim),
                nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            ]))
            dim *= 2

        self.neck_layer = AbunBlock(in_dim=dim, out_dim=dim, group=stage_dim)

        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim * 2, dim, stride=2, kernel_size=2, padding=0, output_padding=0, bias=False),
                AbunBlock(in_dim=dim, out_dim=dim // 2, group=stage_dim),
            ]))
            dim = dim // 2

        self.mapping = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        x_in = self.conv(x_in)
        x = x_in
        encoder = []
        for i, (block, pooling) in enumerate(self.encoder_layers):
            x = block(x)
            x = pooling(x)
            encoder.append(x)
        x = self.neck_layer(x)
        for i, (rpooling, block) in enumerate(self.decoder_layers):
            x = torch.cat([x, encoder[-(i+1)]], dim=1)
            x = rpooling(x)
            x = block(x)
        x_out = x + x_in
        x_out = self.mapping(x_out)
        #x_out = self.softmax(x_out)
        return x_out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out


class PMM(nn.Module):
    def __init__(self, in_dim=3, out_dim=48, spec_num=345, stage=3, dim=32):
        super(PMM, self).__init__()
        self.out_dim = out_dim
        self.Abund_G = Swin(in_dim=in_dim, out_dim=spec_num, stage=stage, dim=dim)
        self.conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, sr):
        B, in_dim, h_dim1, w_dim1 = x.shape
        out_dim, _ = sr.shape
        hb, wb = 64, 64
        pad_h = (hb - (h_dim1 % hb)) % hb  # Prevent the Carrying
        pad_w = (wb - (w_dim1 % wb)) % wb  # Prevent the Carrying
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        abund = self.Abund_G(x)
        out_abund = abund
        B, C, W, H = abund.shape
        abund = torch.reshape(abund, [B, C, W * H])
        x = sr @ abund
        x = torch.reshape(x, [B, self.out_dim, W, H])
        x = self.conv(x)
        return x.contiguous()[:, :, :h_dim1, :w_dim1]


def main():
    model = PMM(in_dim=3, out_dim=48, spec_num=345, stage=3)
    #data1, data2 = torch.rand(1, 31, 16, 16), torch.rand(1, 128, 16, 16)
    data = torch.rand(1, 3, 65, 64)
    sr = torch.rand(48, 345)
    out = model(data, sr)
    hh = 1

if __name__ == "__main__":
    main()


