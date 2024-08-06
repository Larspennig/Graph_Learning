from torch_geometric.data import Dataset, DataLoader, Data
import torch
import os
import numpy as np
import torch_geometric as tg
from sklearn.neighbors import kneighbors_graph
import torch_geometric as tgg
import yaml
import h5py
import open3d as o3d
import torch_geometric.transforms as tgt


class SNpart_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', pre_transform=None, pre_filter=None):
        self.split = split
        self.root = root
        self.transforms = tgt.Compose([tgt.RandomJitter(0.01),
                                      tgt.RandomRotate(30, axis=0),
                                      tgt.RandomRotate(30, axis=1),
                                      tgt.RandomRotate(30, axis=2)])

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root+'/raw/'):
            for filename in filenames:
                full_path = os.path.join(dirpath.split('/')[-1], filename)
                files.append(full_path)
        return files

    @property
    def processed_file_names(self):
        obj_list = []
        if os.path.exists(self.root+'/train_test_split/shuffled_'+self.split+'_file_list.txt'):
            with open(self.root+'/train_test_split/shuffled_'+self.split+'_file_list.txt', 'r') as f:
                obj_list = [line.strip() for line in f.readlines()]
        return obj_list

    def process(self):
        idx = 0
        train_list = []
        val_list = []
        test_list = []

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.

            raw_out = np.loadtxt(raw_path)

            # randomly sample 2048 points
            if raw_out.shape[0] > 2048:
                raw_out = raw_out[np.random.choice(
                    raw_out.shape[0], 2048, replace=False), :]
            else:
                print('Not enough points in ', raw_path)
                print(f'{raw_out.shape[0]} points found')
                idx = idx+1

            pc = Data(x=torch.Tensor(raw_out[:, 3:6]),
                      pos=torch.Tensor(raw_out[:, :3]),
                      y=torch.Tensor(raw_out[:, 6]))
            torch.save(pc, raw_path.replace(
                'raw', 'processed').replace('.txt', '.pt'))
        print('Faulty files number ', idx, ' files')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_dir+'/' +
                          self.processed_file_names[idx])

        data = self.transforms(data)
        return data
