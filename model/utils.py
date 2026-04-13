
import torch
import numpy as np


def set_traj_base_dim(traj_type, feat_dim, vec_dim):
    if traj_type == 'poly_fourier':
        traj_base_dim = 3
        extend_dim = feat_dim * vec_dim * traj_base_dim
    elif traj_type == 'poly':
        traj_base_dim = 0
        extend_dim = feat_dim * vec_dim
    elif traj_type == 'fourier':
        traj_base_dim = 2
        extend_dim = feat_dim * vec_dim * traj_base_dim
    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")
        
    return traj_base_dim, extend_dim


def get_knn(pos: torch.Tensor, k: int = 3, device="cuda"):
    # 1. 차원 체크 및 처리
    # pos의 shape이 [Batch, N, 3]인 경우, 첫 번째 배치([N, 3])만 사용합니다.
    if pos.dim() == 3:
        pos = pos[0]
        
    # 2. Numpy 변환
    pos_np = pos.detach().cpu().numpy()
    
    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    # 3. KNN 모델 빌드 (2차원 데이터 pos_np: [N, 3])
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, 
        algorithm="auto", 
        metric="euclidean"
    ).fit(pos_np)
    
    distances, indices = nn_model.kneighbors(pos_np)
    
    # 자기 자신을 제외한 가장 가까운 k개의 이웃 정보 추출
    distances = distances[:, 1:].astype(np.float32)
    distances = torch.from_numpy(distances).to(device)
    indices = indices[:, 1:].astype(np.int32)
    indices = torch.from_numpy(indices).to(device)
    
    return distances, indices