from torch_geometric.data import Dataset, DataLoader, Data
import torch
import os
import numpy as np
import torch_geometric as tg
from sklearn.neighbors import kneighbors_graph
import open3d as o3d
import yaml

from torch_geometric.io import read_off


class Modelnet40(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train', classes_yml=None):
        self.split = split
        with open(classes_yml, 'r') as f:
            self.classes = yaml.safe_load(f)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        obj_list = []

        for path, subdirs, files in os.walk(self.root+'/'+'raw'):
            for name in files:
                obj_list.append(os.path.join(path.split(sep='/')[-1], name))

        return obj_list

    @property
    def processed_file_names(self):
        obj_list = []

        with open(self.root+'/'+'modelnet40_'+self.split+'.txt', 'r') as f:
            split_list = f.read().split('\n')

        for path, subdirs, files in os.walk(self.root+'/'+'processed'):
            for name in files:
                if name[:-3] in split_list:
                    obj_list.append(os.path.join(
                        path.split(sep='/')[-1], name))

        # Return train or test list
        return obj_list

    def process(self):
        count = 0
        for raw_path in self.raw_paths:

            # Read data from raw_path and convert to point cloud
            raw_points = np.loadtxt(raw_path, delimiter=',')
            normals = raw_points[:, 3:]
            pos = raw_points[:, :3]

            # Read class label
            label = self.classes[raw_path.split(sep='/')[-2]]

            # Sample n=1024 points uniformly out of given 10k points
            idx = np.random.choice(
                np.arange(raw_points.shape[0]), replace=False, size=1024)
            idx.sort()

            # Create torch_geometric data and save
            data = tg.data.Data()
            data.x = torch.tensor(normals[idx]).float()
            data.pos = torch.tensor(pos[idx]).float()
            data.y = torch.tensor(label).float()

            # Transformations
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save Data CHANGE NAME STORED
            save_path = self.processed_dir+'/' + \
                raw_path.split(sep='/')[-2]+'/' + \
                raw_path.split(sep='/')[-1][:-4]+'.pt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(data, save_path)

            count += 1
            print(f'Processed {count}/{len(self.raw_paths)}')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_dir+'/' +
                          self.processed_file_names[idx])
        return data
