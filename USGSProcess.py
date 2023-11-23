import numpy as np
import os
import os.path as osp
import h5py
import hdf5storage

def main():
    database_path = "/home/data/duanshiyao/USGS_raw/ASCIIdata/ASCIIdata_splib07b_cvAVIRISc2014"
    database_lists = ["ChapterA_ArtificialMaterials", "ChapterC_Coatings", "ChapterL_Liquids",
                      "ChapterS_SoilsAndMixtures", "ChapterV_Vegetation"]
    Library_path = "/home/data/duanshiyao/USGS_Library/removed.mat"
    all_SR = []

    for i in range(len(database_lists)):
        full_path = osp.join(database_path, database_lists[i])
        all_lists = os.listdir(full_path)
        all_lists.sort()
        for j in range(len(all_lists)):
            flag = 0 # Charge the flag
            EM_list = []
            with open(osp.join(full_path, all_lists[j]), 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                data = np.float16(line)
                if data == -np.inf:
                    flag = 1
                    break
                EM_list.append(data)
            if flag == 0:
                EM_list = np.array(EM_list)
                all_SR.append(EM_list)
    all_SR = np.float32(np.array(all_SR))
    hdf5storage.savemat(Library_path, {'SR': all_SR}, format='7.3')
    hh = 1


if __name__ == "__main__":
    main()