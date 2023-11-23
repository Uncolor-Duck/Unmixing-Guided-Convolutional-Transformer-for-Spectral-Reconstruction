import torch
from .BMM import BMM
from .PDASS import pdass
from .MST_Plus_Plus import MST_Plus_Plus
from .swin import PMM
from .HSCNN_Plus import HSCNN_Plus
from .AWAN import AWAN
from .Restormer import Restormer
from .ablation import ablation
from .hrnet import SGN

def model_generator(method, pretrained_model_path=None):
    if method == 'BMM':
        model = BMM().cuda()
    elif method == 'pdass':
        model = pdass().cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    elif method == 'Ours':
        model = PMM().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'awan':
        model = AWAN().cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'ablation':
        model = ablation().cuda()
    elif method == 'hrnet':
        model = SGN().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
