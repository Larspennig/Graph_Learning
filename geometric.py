import os
import torch_geometric as tg
import torch
import numpy as np
import open3d as o3d


class Extract_Geometric():
    def __init__(self, data: tg.data.Data):
        self.data = data

    def load_to_open3d(self):
        points = self.data.pos.numpy()
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        return

    def compute_iss_keypoints(self):
        # Extract normals
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(self.pcd)
        return keypoints

    def compute_edges(self):
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        normals = np.asarray(self.pcd.normals)

        return
