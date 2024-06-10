from torch_geometric.data import Dataset, DataLoader, Data
import torch
import os
import numpy as np
import torch_geometric as tg
from sklearn.neighbors import kneighbors_graph
import torch_geometric as tgg
import yaml


class Stanford_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', pre_transform=None, pre_filter=None):
        self.split = split
        with open('classes_seg.yml', 'r') as f:
            self.classes = yaml.safe_load(f)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        area_list = []
        obj_list = []

        for area in os.listdir(self.root+'/raw/'):
            area_list.append(area)
            for obj in os.listdir(self.root+'/raw'+'/'+area):
                if obj == '.DS_Store':
                    continue
                obj_list.append(area+'/'+obj)

        return obj_list

    @property
    def processed_file_names(self):
        area_list = []
        obj_list = []

        for area in os.listdir(self.root+'/processed/'):
            if self.split == 'test':
                if area != 'Area_5':
                    continue
            area_list.append(area)
            for obj in os.listdir(self.root+'/processed/'+area):
                obj_list.append(area+'/'+obj)

        return obj_list

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            obj_list = next(os.walk(raw_path+'/Annotations'))[2]

            labels = []

            point_cloud = np.empty((0, 6))

            for obj in obj_list:
                if obj == '.DS_Store':
                    continue
                if obj == 'Icon':
                    continue

                # load point_cloud and labels
                obj_array = np.loadtxt(raw_path+f'/Annotations/{obj}',
                                       delimiter=' ')

                point_cloud = np.concatenate([point_cloud, obj_array])
                labels = labels + [f"{obj.split(sep='_')[0]}"] * len(obj_array)

            # transform labels
            labels = torch.tensor([self.classes[label] for label in labels])

            data = Data(x=torch.from_numpy(point_cloud[:, 3:]),
                        pos=torch.from_numpy(point_cloud[:, :3]),
                        y=labels)

            data = tg.transforms.GridSampling(0.03)(data)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save pointcloud
            torch.save(data, self.processed_dir+'/'+raw_path.split(sep='/')
                       [-2]+'/'+raw_path.split(sep='/')[-1]+'.pt')
            idx += 1
            print(f'Processed {idx}/{len(self.raw_paths)}')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_dir+'/' +
                          self.processed_file_names[idx])
        return data
