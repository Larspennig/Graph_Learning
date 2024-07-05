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


class SNpart_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', pre_transform=None, pre_filter=None):
        self.split = split
        with open(root+'/raw/'+'all_object_categories.txt', 'r') as f:
            lines = f.readlines()
        self.item_dict = {line.split()[0]: int(line.split()[1]) for line in lines}
        
        self.transform_1 = tg.transforms.RandomJitter(translate=0.0001)
        self.transform_2 = tg.transforms.RandomRotate(180, axis=2)

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        files = []
        with open(self.root+'/raw/'+'train_hdf5_file_list.txt', 'r') as f:
            files = files +[line.strip() for line in f.readlines()]
        with open(self.root+'/raw/'+'val_hdf5_file_list.txt', 'r') as f:
            files = files + [line.strip() for line in f.readlines()]
        with open(self.root+'/raw/'+'test_hdf5_file_list.txt', 'r') as f:
            files = files + [line.strip() for line in f.readlines()]
        return files

    @property
    def processed_file_names(self):
        obj_list = []
        if os.path.exists(self.root+'/'+self.split+'_list.txt'):
            with open(self.root+'/'+self.split+'_list.txt', 'r') as f:
                obj_list = [line.strip() for line in f.readlines()]

        return obj_list

    def process(self):
        idx = 0
        train_list = []
        val_list = []
        test_list = []

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with h5py.File(raw_path) as f:
                data = f['data'][:]
                labels = f['label'][:]
                pid = f['pid'][:]

            for i in range(data.shape[0]):
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[i,:,:]))
                cloud.estimate_normals()
                normals = torch.Tensor(np.array(cloud.normals))

                pos = torch.Tensor(np.array(data[i,:,:]))
                # normalize pos
                pos = pos - pos.mean(dim=0, keepdim=True)
                pos = pos / torch.norm(pos, dim=0).max()

                pc = Data(x=normals, pos = pos, y=torch.Tensor(pid[i,:]))
                obj = torch.Tensor(labels[i])
                torch.save((pc,obj), self.processed_dir+'/'+'part_'+str(idx).zfill(5)+'.pt')

                if raw_path.split('.')[0].split('_')[-1][:-1] == 'train':
                    train_list.append('part_'+str(idx).zfill(5)+'.pt')
                elif raw_path.split('.')[0].split('_')[-1][:-1]== 'val':
                    val_list.append('part_'+str(idx).zfill(5)+'.pt')
                elif raw_path.split('.')[0].split('_')[-1][:-1] == 'test':
                    test_list.append('part_'+str(idx).zfill(5)+'.pt')


                idx += 1
            
            print(f'Processed files: {idx}')

        # save train, val, test lists
        with open(self.root+'/train_list.txt', 'w') as f:
            for item in train_list:
                f.write("%s\n" % item)
        with open(self.root+'/val_list.txt', 'w') as f:
            for item in val_list:
                f.write("%s\n" % item)
        with open(self.root+'/test_list.txt', 'w') as f:
            for item in test_list:
                f.write("%s\n" % item)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_dir+'/' +
                          self.processed_file_names[idx])
        
        data = self.transform_1(data)
        data = self.transform_2(data)
        return data[0]
