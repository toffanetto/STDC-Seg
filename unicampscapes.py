import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np

class UnicampScapes(torch.utils.data.Dataset):
    """UnicampScapes is a dataset with CityScapes-like imagens taken in Unicamp campus"""
    def __init__(self, rootpth):
        super(UnicampScapes, self).__init__()
        
        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'unicamp_scapes', 'inference')
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))
            
        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, idx):
        _ = 0
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        img = Image.open(impth).convert('RGB')
        img_raw = Image.open(impth)
        img_raw.load()
        img_raw = np.asarray(img_raw, dtype="int32")
        img = self.to_tensor(img)
        
        return img, _, fn, img_raw

    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    from tqdm import tqdm
    ds = UnicampScapes('./data/')