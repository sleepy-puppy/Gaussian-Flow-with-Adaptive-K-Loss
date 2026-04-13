### 원본 ###
# import polyfourier
# import numpy as np

# import roma
# import torch
# import torch.nn as nn

# from dataclasses import dataclass
# from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
# from pointrix.model.point_cloud import POINTSCLOUD_REGISTRY

# from .utils import set_traj_base_dim, get_knn

# @POINTSCLOUD_REGISTRY.register()
# class GaussianFlowPointCloud(GaussianPointCloud):
#     @dataclass
#     class Config(GaussianPointCloud.Config):
#         pos_traj_type: str = 'poly_fourier'
#         pos_traj_dim: int = 3
#         rot_traj_type: str = 'poly_fourier'
#         rot_traj_dim: int = 3
        
#         feat_traj_type: str = 'poly_fourier'
#         feat_traj_dim: int = 3
        
#         rescale_t: bool = True
#         rescale_value: float = 1.0
        
#         offset_t: bool = True
#         offset_value: float = 0.0
        
#         normliaze_rot: bool = False
#         normalize_timestamp: bool = False
        
#         random_noise: bool = False
#         max_steps: int = 0
        
#     cfg: Config

#     def setup(self, point_cloud=None):
#         super().setup(point_cloud)
        
#         self.rot_traj_base_dim, rot_extend_dim = set_traj_base_dim(
#             self.cfg.rot_traj_type, self.cfg.rot_traj_dim, 4
#         )
            
#         # rots = torch.zeros((len(self), 4+rot_extend_dim))
#         # rots[:, 0] = 1
#         # self.rotation = nn.Parameter(
#         #     rots.contiguous().requires_grad_(True)
#         # )
#         self.register_atribute(
#             "rot_params", 
#             torch.zeros((len(self), self.cfg.rot_traj_dim, 4, self.rot_traj_base_dim)),
#             # torch.zeros((len(self), rot_extend_dim))
#         )
            
#         self.rot_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.rot_traj_type
#         )
        
#         # init position trajectory
#         self.pos_traj_base_dim, pos_extend_dim = set_traj_base_dim(
#             self.cfg.pos_traj_type, self.cfg.pos_traj_dim, 3
#         )
            
#         # self.position = nn.Parameter(
#         #     torch.cat([
#         #         self.position,
#         #         torch.zeros(
#         #             (len(self), pos_extend_dim),
#         #             dtype=torch.float32
#         #         )
#         #     ], dim=1).contiguous().requires_grad_(True)
#         # )
#         self.register_atribute(
#             "pos_params", 
#             torch.zeros((len(self), self.cfg.pos_traj_dim, 3, self.pos_traj_base_dim)),
#             # torch.zeros((len(self), pos_extend_dim))
#         )
#         self.pos_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.pos_traj_type
#         )
        
#         self.feat_traj_base_dim, feat_extend_dim = set_traj_base_dim(
#             self.cfg.feat_traj_type, self.cfg.feat_traj_dim, 3
#         )
        
#         self.register_atribute(
#             "feat_params", 
#             torch.zeros((len(self), self.cfg.feat_traj_dim, 3, self.feat_traj_base_dim)),
#             # torch.zeros((len(self), feat_extend_dim))
#         )
#         self.feat_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.feat_traj_type
#         )
        
#         self.register_atribute("time_center", torch.zeros((len(self), 1)))
        
#     @torch.no_grad()
#     def gen_knn(self):
#         self.set_timestep(0.)
#         theta_w = 100_000
#         self.knn_distances_0, self.knn_indices_0 = get_knn(
#             self.get_position_flow, k=20
#         )
#         self.knn_weights_0 = torch.exp(-theta_w*torch.pow(self.knn_distances_0, 2))

#     def knn_loss(self, t):
#         timestamp = self.make_time_features(t)
#         if t == self.max_timestamp:
#             t1 = timestamp-self.offset_width
#             t2 = timestamp
#         else:
#             t1 = timestamp
#             t2 = timestamp+self.offset_width
            
#         self.fwd_flow(t1)
#         t1_pos = self.get_position_flow
#         t1_rot = self.get_rotation_flow
#         self.fwd_flow(t2)
#         t2_pos = self.get_position_flow
#         t2_rot = self.get_rotation_flow
        
#         t1_dist = t1_pos[self.knn_indices_0] - t1_pos.unsqueeze(1)
#         t2_dist = t2_pos[self.knn_indices_0] - t2_pos.unsqueeze(1)
        
#         R1 = roma.unitquat_to_rotmat(t1_rot)
#         R2 = roma.unitquat_to_rotmat(t2_rot)
#         R = R1 @ R2.inverse()
        
#         dist = (t1_dist - (R @ t2_dist)) ** 2
#         loss = (self.knn_weights_0.unsqueeze(-1) * dist).mean()
#         return loss
        
#     def make_time_features(self, t, training=False, training_step=0):
#         # if isinstance(t, torch.Tensor):
#         #     t = t.item()
            
#         if self.cfg.normalize_timestamp:
#             self.timestamp = t / self.max_timestamp
#             self.offset_width = (1/self.max_frames)*0.1
#         else:
#             self.timestamp = t
#             self.offset_width = 0.01
            
#         if self.cfg.rescale_t:
#             self.timestamp *= self.cfg.rescale_value
#             self.offset_width *= self.cfg.rescale_value
            
#         if self.cfg.offset_t:
#             self.timestamp += self.cfg.offset_value
            
#         if self.cfg.random_noise and training:
#             noise_weight = self.offset_width * (1 - (training_step/self.cfg.max_steps))
#             self.timestamp += noise_weight*torch.randn_like(self.timestamp)
            
#         if isinstance(t, float):
#             return self.timestamp - self.time_center[None, :, 0]
            
#         return self.timestamp[:, None] - self.time_center[None, :, 0]
    
#     def fwd_flow(self, timestamp_batch):
#         pos_base = self.position[:, :3]
#         rot_base = self.rotation[:, :4]
        
#         position_flow = []
#         rotation_flow = []
#         feat_flow = []
        
#         for i in range(timestamp_batch.size(0)):
#             timestamp = timestamp_batch[i, :, None]
#             pos_traj = self.pos_fit_model(
#                 self.pos_params, 
#                 # pos_traj_params,
#                 timestamp, 
#                 self.cfg.pos_traj_dim,
#             )
#             position_flow.append(pos_base + pos_traj)
#             rot_traj = self.rot_fit_model(
#                 # rot_traj_params, 
#                 self.rot_params,
#                 timestamp, 
#                 self.cfg.rot_traj_dim,
#             )
#             rotation_flow.append(rot_base + rot_traj)
            
            
#             feat_traj = self.feat_fit_model(
#                 self.feat_params, 
#                 timestamp, 
#                 self.cfg.feat_traj_dim,
#             )
#             feat_flow.append(self.features + feat_traj.unsqueeze(1))
            
#         self.position_flow = torch.stack(position_flow, dim=0)
#         self.rotation_flow = torch.stack(rotation_flow, dim=0)
#         self.feat_flow = torch.stack(feat_flow, dim=0)
        
#     def set_timestep(self, t, training=False, training_step=0):
#         self.t = t
#         timestamp = self.make_time_features(t, training, training_step)
#         self.fwd_flow(timestamp)

#     @property
#     def get_rotation_flow(self):
#         return self.rotation_activation(self.rotation_flow)

#     @property
#     def get_position_flow(self):
#         return self.position_flow
    
#     @property
#     def get_rotation(self):
#         return self.rotation_activation(self.rotation[:, :4])
    
#     @property
#     def get_position(self):
#         return self.position[:, :3]
    
#     @property
#     def get_shs_flow(self):
#         batch_feat_rest = self.features_rest[None].expand(
#             self.position_flow.size(0), -1, -1, -1
#         )
#         return torch.cat([
#             self.feat_flow, batch_feat_rest,
#         ], dim=2)



# ### k loss 실행되도록 수정 ###
# import polyfourier
# import numpy as np

# import roma
# import torch
# import torch.nn as nn

# from dataclasses import dataclass
# from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
# from pointrix.model.point_cloud import POINTSCLOUD_REGISTRY

# from .utils import set_traj_base_dim, get_knn

# @POINTSCLOUD_REGISTRY.register()
# class GaussianFlowPointCloud(GaussianPointCloud):
#     @dataclass
#     class Config(GaussianPointCloud.Config):
#         pos_traj_type: str = 'poly_fourier'
#         pos_traj_dim: int = 3
#         rot_traj_type: str = 'poly_fourier'
#         rot_traj_dim: int = 3
        
#         feat_traj_type: str = 'poly_fourier'
#         feat_traj_dim: int = 3
        
#         rescale_t: bool = True
#         rescale_value: float = 1.0
        
#         offset_t: bool = True
#         offset_value: float = 0.0
        
#         normliaze_rot: bool = False
#         normalize_timestamp: bool = False
        
#         random_noise: bool = False
#         max_steps: int = 0
        
#     cfg: Config

#     def setup(self, point_cloud=None):
#         super().setup(point_cloud)
        
#         self.rot_traj_base_dim, rot_extend_dim = set_traj_base_dim(
#             self.cfg.rot_traj_type, self.cfg.rot_traj_dim, 4
#         )
            
#         # rots = torch.zeros((len(self), 4+rot_extend_dim))
#         # rots[:, 0] = 1
#         # self.rotation = nn.Parameter(
#         #     rots.contiguous().requires_grad_(True)
#         # )
#         self.register_atribute(
#             "rot_params", 
#             torch.zeros((len(self), self.cfg.rot_traj_dim, 4, self.rot_traj_base_dim)),
#             # torch.zeros((len(self), rot_extend_dim))
#         )
            
#         self.rot_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.rot_traj_type
#         )
        
#         # init position trajectory
#         self.pos_traj_base_dim, pos_extend_dim = set_traj_base_dim(
#             self.cfg.pos_traj_type, self.cfg.pos_traj_dim, 3
#         )
            
#         # self.position = nn.Parameter(
#         #     torch.cat([
#         #         self.position,
#         #         torch.zeros(
#         #             (len(self), pos_extend_dim),
#         #             dtype=torch.float32
#         #         )
#         #     ], dim=1).contiguous().requires_grad_(True)
#         # )
#         self.register_atribute(
#             "pos_params", 
#             torch.zeros((len(self), self.cfg.pos_traj_dim, 3, self.pos_traj_base_dim)),
#             # torch.zeros((len(self), pos_extend_dim))
#         )
#         self.pos_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.pos_traj_type
#         )
        
#         self.feat_traj_base_dim, feat_extend_dim = set_traj_base_dim(
#             self.cfg.feat_traj_type, self.cfg.feat_traj_dim, 3
#         )
        
#         self.register_atribute(
#             "feat_params", 
#             torch.zeros((len(self), self.cfg.feat_traj_dim, 3, self.feat_traj_base_dim)),
#             # torch.zeros((len(self), feat_extend_dim))
#         )
#         self.feat_fit_model = polyfourier.get_fit_model(
#             type_name=self.cfg.feat_traj_type
#         )
        
#         self.register_atribute("time_center", torch.zeros((len(self), 1)))
        
#         self.max_timestamp = 1.0
        
#     @torch.no_grad()
#     def gen_knn(self):
#         self.set_timestep(0.)
#         theta_w = 100_000
#         self.knn_distances_0, self.knn_indices_0 = get_knn(
#             self.get_position_flow, k=40
#         )
#         self.knn_weights_0 = torch.exp(-theta_w*torch.pow(self.knn_distances_0, 2))

#     def knn_loss(self, t):
#         timestamp = self.make_time_features(t)
        
#         # 1. 배치 차원 중 하나라도 끝인지 확인
#         is_last_frame = (t >= self.max_timestamp).any()
        
#         if is_last_frame:
#             t1 = timestamp - self.offset_width
#             t2 = timestamp
#         else:
#             t1 = timestamp
#             t2 = timestamp + self.offset_width
            
#         self.fwd_flow(t1)
#         # .squeeze(0)를 사용하여 [1, N, 3] -> [N, 3]으로 만듭니다.
#         t1_pos = self.get_position_flow.squeeze(0) 
#         t1_rot = self.get_rotation_flow.squeeze(0)
        
#         self.fwd_flow(t2)
#         t2_pos = self.get_position_flow.squeeze(0)
#         t2_rot = self.get_rotation_flow.squeeze(0)
        
#         # 2. [N, K, 3] 형태의 연산을 수행 (배치 차원이 없으므로 훨씬 가벼움)
#         # t1_pos[self.knn_indices_0] -> [N, K, 3]
#         # t1_pos.unsqueeze(1) -> [N, 1, 3]
#         t1_dist = t1_pos[self.knn_indices_0.long()] - t1_pos.unsqueeze(1)
#         t2_dist = t2_pos[self.knn_indices_0.long()] - t2_pos.unsqueeze(1)
        
#         # 3. 회전 행렬 연산
#         R1 = roma.unitquat_to_rotmat(t1_rot)
#         R2 = roma.unitquat_to_rotmat(t2_rot)
#         # R2.inverse() 대신 R2.transpose(-1, -2)가 메모리와 속도면에서 효율적입니다 (회전행렬 특성상)
#         R = R1 @ R2.transpose(-1, -2) 
        
#         # 4. 거리 계산 (R @ t2_dist 연산을 위해 차원 맞춤)
#         # R: [N, 3, 3], t2_dist: [N, K, 3] -> R @ t2_dist.transpose(-1, -2)
#         rotated_t2_dist = (R.unsqueeze(1) @ t2_dist.unsqueeze(-1)).squeeze(-1)
        
#         dist = (t1_dist - rotated_t2_dist) ** 2
#         loss = (self.knn_weights_0.unsqueeze(-1) * dist).mean()
        
#         return loss
        
#     def make_time_features(self, t, training=False, training_step=0):
#         # if isinstance(t, torch.Tensor):
#         #     t = t.item()
            
#         if self.cfg.normalize_timestamp:
#             self.timestamp = t / self.max_timestamp
#             self.offset_width = (1/self.max_frames)*0.1
#         else:
#             self.timestamp = t
#             self.offset_width = 0.01
            
#         if self.cfg.rescale_t:
#             self.timestamp *= self.cfg.rescale_value
#             self.offset_width *= self.cfg.rescale_value
            
#         if self.cfg.offset_t:
#             self.timestamp += self.cfg.offset_value
            
#         if self.cfg.random_noise and training:
#             noise_weight = self.offset_width * (1 - (training_step/self.cfg.max_steps))
#             self.timestamp += noise_weight*torch.randn_like(self.timestamp)
            
#         if isinstance(t, float):
#             return self.timestamp - self.time_center[None, :, 0]
            
#         return self.timestamp[:, None] - self.time_center[None, :, 0]
    
#     def fwd_flow(self, timestamp_batch):
#         pos_base = self.position[:, :3]
#         rot_base = self.rotation[:, :4]
        
#         position_flow = []
#         rotation_flow = []
#         feat_flow = []
        
#         for i in range(timestamp_batch.size(0)):
#             timestamp = timestamp_batch[i, :, None]
#             pos_traj = self.pos_fit_model(
#                 self.pos_params, 
#                 # pos_traj_params,
#                 timestamp, 
#                 self.cfg.pos_traj_dim,
#             )
#             position_flow.append(pos_base + pos_traj)
#             rot_traj = self.rot_fit_model(
#                 # rot_traj_params, 
#                 self.rot_params,
#                 timestamp, 
#                 self.cfg.rot_traj_dim,
#             )
#             rotation_flow.append(rot_base + rot_traj)
            
            
#             feat_traj = self.feat_fit_model(
#                 self.feat_params, 
#                 timestamp, 
#                 self.cfg.feat_traj_dim,
#             )
#             feat_flow.append(self.features + feat_traj.unsqueeze(1))
            
#         self.position_flow = torch.stack(position_flow, dim=0)
#         self.rotation_flow = torch.stack(rotation_flow, dim=0)
#         self.feat_flow = torch.stack(feat_flow, dim=0)
        
#     def set_timestep(self, t, training=False, training_step=0):
#         self.t = t
#         timestamp = self.make_time_features(t, training, training_step)
#         self.fwd_flow(timestamp)

#     @property
#     def get_rotation_flow(self):
#         return self.rotation_activation(self.rotation_flow)

#     @property
#     def get_position_flow(self):
#         return self.position_flow
    
#     @property
#     def get_rotation(self):
#         return self.rotation_activation(self.rotation[:, :4])
    
#     @property
#     def get_position(self):
#         return self.position[:, :3]
    
#     @property
#     def get_shs_flow(self):
#         batch_feat_rest = self.features_rest[None].expand(
#             self.position_flow.size(0), -1, -1, -1
#         )
#         return torch.cat([
#             self.feat_flow, batch_feat_rest,
#         ], dim=2)




### knn 수정 ###
import polyfourier
import numpy as np

import roma
import torch
import torch.nn as nn

from dataclasses import dataclass
from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from pointrix.model.point_cloud import POINTSCLOUD_REGISTRY

from .utils import set_traj_base_dim, get_knn

# 아주 작은 값 epsilon 정의
EPSILON = 1e-6 

@POINTSCLOUD_REGISTRY.register()
class GaussianFlowPointCloud(GaussianPointCloud):
    @dataclass
    class Config(GaussianPointCloud.Config):
        pos_traj_type: str = 'poly_fourier'
        pos_traj_dim: int = 3
        rot_traj_type: str = 'poly_fourier'
        rot_traj_dim: int = 3
        
        feat_traj_type: str = 'poly_fourier'
        feat_traj_dim: int = 3
        
        rescale_t: bool = True
        rescale_value: float = 1.0
        
        offset_t: bool = True
        offset_value: float = 0.0
        
        normliaze_rot: bool = False
        normalize_timestamp: bool = False
        
        random_noise: bool = False
        max_steps: int = 0
        
        # 코사인 유사도 하이퍼파라미터
        alpha_for_cosine_sim: float = 10
        cosine_sim_threshold: float = 0
        
    cfg: Config

    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        
        self.rot_traj_base_dim, rot_extend_dim = set_traj_base_dim(
            self.cfg.rot_traj_type, self.cfg.rot_traj_dim, 4
        )
            
        self.register_atribute(
            "rot_params", 
            torch.zeros((len(self), self.cfg.rot_traj_dim, 4, self.rot_traj_base_dim)),
        )
            
        self.rot_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.rot_traj_type
        )
        
        # init position trajectory
        self.pos_traj_base_dim, pos_extend_dim = set_traj_base_dim(
            self.cfg.pos_traj_type, self.cfg.pos_traj_dim, 3
        )
            
        self.register_atribute(
            "pos_params", 
            torch.zeros((len(self), self.cfg.pos_traj_dim, 3, self.pos_traj_base_dim)),
        )
        self.pos_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.pos_traj_type
        )
        
        self.feat_traj_base_dim, feat_extend_dim = set_traj_base_dim(
            self.cfg.feat_traj_type, self.cfg.feat_traj_dim, 3
        )
        
        self.register_atribute(
            "feat_params", 
            torch.zeros((len(self), self.cfg.feat_traj_dim, 3, self.feat_traj_base_dim)),
        )
        self.feat_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.feat_traj_type
        )
        
        self.register_atribute("time_center", torch.zeros((len(self), 1)))
        
        self.max_timestamp = 1.0

    def _get_differentiated_params(self):
        """
        pos_params(N, J, D, K)를 미분하여 반환
        """
        pos_params_copy = self.pos_params.data.clone()
        pos_params_grad = torch.zeros_like(pos_params_copy)
        
        # J: 1 ~ (pos_traj_dim-1) 차수 미분
        for j in range(1, self.cfg.pos_traj_dim): 
            # Poly
            pos_params_grad[:, j, :, 0] = j * pos_params_copy[:, j, :, 0]
            # Sin
            pos_params_grad[:, j, :, 1] = 2 * np.pi * j * pos_params_copy[:, j, :, 1]
            # Cos
            pos_params_grad[:, j, :, 2] = 2 * np.pi * j * pos_params_copy[:, j, :, 2]
            
        return pos_params_grad

    def _normalize_params_grad(self, pos_params_grad):
        """
        미분 계수 정규화 (N 차원에 대해 평균/표준편차 계산)
        """
        pos_params_mean = pos_params_grad.mean(dim=0, keepdim=True)
        pos_params_std = pos_params_grad.std(dim=0, keepdim=True)
        
        pos_params_norm = (pos_params_grad - pos_params_mean) / (pos_params_std + EPSILON)
        pos_params_norm[:, 0, :, :] = 0 # 0차는 0으로 고정
        
        return pos_params_norm

    def _get_similarity_mask_vectorized(self, pos_params_norm):
        """
        모든 가우시안과 그 이웃들 간의 코사인 유사도를 '한 번에(Vectorized)' 계산하여 
        마스크를 반환합니다. for문을 사용하지 않습니다.
        """
        N = pos_params_norm.shape[0]
        K_neighbors = self.knn_indices_0.shape[1]
        
        # 1. 파라미터 평탄화 (Flatten)
        # (N, J, D, K) -> (N, Feature_Dim)
        # 내적 계산을 쉽게 하기 위해 한 줄로 폅니다.
        flat_params = pos_params_norm.view(N, -1)
        feature_dim = flat_params.shape[1]

        # 2. 주파수 가중치 벡터 생성 및 적용
        j_dims = self.cfg.pos_traj_dim
        d_dims = 3
        k_dims = self.pos_traj_base_dim
        
        # (J, D, K) 형태의 주파수 인덱스 생성
        freq_indices = torch.arange(j_dims, device=pos_params_norm.device)\
            .view(j_dims, 1, 1).repeat(1, d_dims, k_dims).flatten()
            
        # 가중치 계산: w = 1 / (1 + alpha * freq^2)
        alpha = self.cfg.alpha_for_cosine_sim
        weights = 1.0 / (1.0 + alpha * (freq_indices ** 2))
        
        # 0차 주파수 가중치는 0으로
        weights[0 : d_dims * k_dims] = 0
        
        # 가중치 적용 (Broadcasting): (N, F) * (F,) -> (N, F)
        # 미리 가중치를 곱해두면, 이후 내적 계산 시 자동으로 가중 내적이 됩니다.
        weighted_params = flat_params * weights

        # 3. 중심 가우시안(Center)과 이웃 가우시안(Neighbor) 파라미터 준비
        
        # Center: (N, 1, F) - 브로드캐스팅을 위해 차원 추가
        center_feats = weighted_params.unsqueeze(1)
        
        # Neighbor: (N, K, F) - knn_indices_0를 이용해 한 번에 가져옴 (Fancy Indexing)
        # 마치 t1_pos[self.knn_indices_0] 하던 것과 똑같습니다.
        neighbor_feats = weighted_params[self.knn_indices_0] 

        # 4. 코사인 유사도 계산 (Batch 연산)
        
        # Dot Product: (N, 1, F) * (N, K, F) -> (N, K, F) --sum--> (N, K)
        dot_product = (center_feats * neighbor_feats).sum(dim=-1)
        
        # Norm: 각 벡터의 크기 계산
        norm_center = torch.norm(center_feats, dim=-1)     # (N, 1)
        norm_neighbor = torch.norm(neighbor_feats, dim=-1) # (N, K)
        
        # Cosine Similarity: (N, K)
        cosine_sim = dot_product / ((norm_center * norm_neighbor) + EPSILON)
        
        # 5. 마스크 생성
        # Threshold보다 큰 경우 True
        mask = cosine_sim > self.cfg.cosine_sim_threshold
        
        # 자기 자신(Self)이 이웃에 포함되어 있다면 유사도가 1.0이므로 True가 될 것입니다.
        # 강체성 Loss에서 거리(dist)가 0이면 Loss도 0이므로 포함되어도 문제는 없으나,
        # 굳이 제외하고 싶다면 아래 주석 해제:
        # self_mask = self.knn_indices_0 != torch.arange(N, device=pos_params_norm.device).unsqueeze(1)
        # mask = mask & self_mask
        
        return mask

    @torch.no_grad()
    def gen_knn(self):
        self.set_timestep(0.)
        theta_w = 100_000
        self.knn_distances_0, self.knn_indices_0 = get_knn(
            self.get_position_flow, k=40
        )
        self.knn_weights_0 = torch.exp(-theta_w*torch.pow(self.knn_distances_0, 2))

    def knn_loss(self, t):
        # 1. 미분 및 정규화
        pos_params_grad = self._get_differentiated_params()
        pos_params_norm = self._normalize_params_grad(pos_params_grad)

        # 2. 벡터화된 코사인 유사도 마스크 생성 (For문 없음!)
        # valid_neighbor_mask Shape: (N, K)
        valid_neighbor_mask = self._get_similarity_mask_vectorized(pos_params_norm)

        # ----------------------------------------------------
        # 기존 Flow 및 KNN Loss 계산
        # ----------------------------------------------------
        timestamp = self.make_time_features(t)
        if t == self.max_timestamp:
            t1 = timestamp-self.offset_width
            t2 = timestamp
        else:
            t1 = timestamp
            t2 = timestamp+self.offset_width
            
        self.fwd_flow(t1)
        # .squeeze(0)를 추가하여 [1, N, 3] -> [N, 3]으로 확실히 압축합니다.
        t1_pos = self.get_position_flow.squeeze(0) 
        t1_rot = self.get_rotation_flow.squeeze(0)
        
        self.fwd_flow(t2)
        t2_pos = self.get_position_flow.squeeze(0)
        t2_rot = self.get_rotation_flow.squeeze(0)
        
        # 이웃들의 위치 가져오기 (Fancy Indexing) -> (N, K, 3) 수정 전
        # 이제 t1_pos는 [N, 3], self.knn_indices_0는 [N, K]입니다.
        # 결과값 t1_dist는 [N, K, 3]이 되어 메모리가 아주 적게 듭니다.
        t1_dist = t1_pos[self.knn_indices_0.long()] - t1_pos.unsqueeze(1)
        t2_dist = t2_pos[self.knn_indices_0.long()] - t2_pos.unsqueeze(1)
        
        # 회전 행렬 연산 (차원 주의)
        R1 = roma.unitquat_to_rotmat(t1_rot) # [N, 3, 3]
        R2 = roma.unitquat_to_rotmat(t2_rot) # [N, 3, 3]
        R = R1 @ R2.transpose(-1, -2)        # [N, 3, 3]
        
        # 거리 오차(Loss) 계산: (N, K)
        # (N, K, 3) - (N, 3, 3) @ (N, K, 3, 1) -> 차원 맞춤 필요
        # R @ t2_dist: (N, 3, 3) @ (N, K, 3) -> (N, K, 3)로 브로드캐스팅 연산됨
        # 정확히는 (N, 1, 3, 3) @ (N, K, 3, 1) 처럼 동작해야 함.
        # roma 라이브러리나 torch.matmul의 브로드캐스팅 규칙에 따라:
        rotated_dist = (R.unsqueeze(1) @ t2_dist.unsqueeze(-1)).squeeze(-1)
        dist = (t1_dist - rotated_dist) ** 2
        
        # (N, K, 3) -> (N, K) 거리 제곱 합
        dist_loss_map = dist.sum(dim=-1) 
        
        # ----------------------------------------------------
        # 마스크 적용 (Filtering)
        # ----------------------------------------------------
        # 마스크가 True인 이웃들만 가중치를 살리고, 나머지는 0으로 만듦
        masked_weights = self.knn_weights_0 * valid_neighbor_mask.float()
        # print('mask neighbors :', masked_weights)
        
        # 최종 Loss 계산
        loss = (masked_weights * dist_loss_map).sum() / (masked_weights.sum() + EPSILON)
        
        return loss
        
    def make_time_features(self, t, training=False, training_step=0):
        if self.cfg.normalize_timestamp:
            self.timestamp = t / self.max_timestamp
            self.offset_width = (1/self.max_frames)*0.1
        else:
            self.timestamp = t
            self.offset_width = 0.01
            
        if self.cfg.rescale_t:
            self.timestamp *= self.cfg.rescale_value
            self.offset_width *= self.cfg.rescale_value
            
        if self.cfg.offset_t:
            self.timestamp += self.cfg.offset_value
            
        if self.cfg.random_noise and training:
            noise_weight = self.offset_width * (1 - (training_step/self.cfg.max_steps))
            self.timestamp += noise_weight*torch.randn_like(self.timestamp)
            
        if isinstance(t, float):
            return self.timestamp - self.time_center[None, :, 0]
            
        return self.timestamp[:, None] - self.time_center[None, :, 0]
    
    def fwd_flow(self, timestamp_batch):
        pos_base = self.position[:, :3]
        rot_base = self.rotation[:, :4]
        
        position_flow = []
        rotation_flow = []
        feat_flow = []
        
        for i in range(timestamp_batch.size(0)):
            timestamp = timestamp_batch[i, :, None]
            pos_traj = self.pos_fit_model(
                self.pos_params, 
                timestamp, 
                self.cfg.pos_traj_dim,
            )
            position_flow.append(pos_base + pos_traj)
            rot_traj = self.rot_fit_model(
                self.rot_params,
                timestamp, 
                self.cfg.rot_traj_dim,
            )
            rotation_flow.append(rot_base + rot_traj)
            
            feat_traj = self.feat_fit_model(
                self.feat_params, 
                timestamp, 
                self.cfg.feat_traj_dim,
            )
            feat_flow.append(self.features + feat_traj.unsqueeze(1))
            
        self.position_flow = torch.stack(position_flow, dim=0)
        self.rotation_flow = torch.stack(rotation_flow, dim=0)
        self.feat_flow = torch.stack(feat_flow, dim=0)
        
    def set_timestep(self, t, training=False, training_step=0):
        self.t = t
        timestamp = self.make_time_features(t, training, training_step)
        self.fwd_flow(timestamp)

    @property
    def get_rotation_flow(self):
        return self.rotation_activation(self.rotation_flow)

    @property
    def get_position_flow(self):
        return self.position_flow
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation[:, :4])
    
    @property
    def get_position(self):
        return self.position[:, :3]
    
    @property
    def get_shs_flow(self):
        batch_feat_rest = self.features_rest[None].expand(
            self.position_flow.size(0), -1, -1, -1
        )
        return torch.cat([
            self.feat_flow, batch_feat_rest,
        ], dim=2)