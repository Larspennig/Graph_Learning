from torch_geometric.data import Dataset, DataLoader, Data
import torch
import os
import numpy as np
import torch_geometric as tg
from sklearn.neighbors import kneighbors_graph
import torch_geometric as tgg
import yaml
from utils.transform import RandomScale, RandomDropColor

def crop_data(data, N_max):
    if data.num_nodes > N_max:
        # get the indices of the N_max closest points to the initial point 
        init_idx = torch.randint(0, data.num_nodes-1,[1,1]).squeeze()
        dists = ((data.pos - data.pos[init_idx])**2).squeeze().sum(dim=1)
        idx = torch.argsort(dists)[:N_max]

        data.x = data.x[idx]
        data.pos = data.pos[idx]
        data.y = data.y[idx]

    return data


class Stanford_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', N_max = None, pre_transform=None, pre_filter=None):
        self.split = split
        self.N_max = N_max
        with open('classes_seg.yml', 'r') as f:
            self.classes = yaml.safe_load(f)
        self.transform_1 = tg.transforms.RandomJitter(translate=0.01)
        self.transform_2 = tg.transforms.RandomRotate(180, axis=2)
        self.transform_3 = RandomScale(scale_low=0.8, scale_high=1.2)
        self.transform_4 = RandomDropColor(p=0.8, color_augment=0.0)

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
            if self.split == 'train':
                if area == 'Area_5':
                    continue
            if area == 'pre_transform.pt' or area == 'pre_filter.pt':
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

        a = 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_dir+'/' +
                          self.processed_file_names[idx])
        
        if self.N_max is not None:
            # do not crop while testing
            data = crop_data(data, self.N_max)

        # only do data transform if training 
        if self.split == 'train':
            data = self.transform_1(data)
            data = self.transform_2(data)
            data = self.transform_3(data)
            data = self.transform_4(data)

        # add absolute positions to features and turn tensors to float
        data.x = torch.cat([data.x, data.pos], dim=1).float()
        data.pos = data.pos.float()
        return data
