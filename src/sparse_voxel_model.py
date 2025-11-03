# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np

import torch
import svraster_cuda

from src.sparse_voxel_gears.constructor import SVConstructor # initialize the model
from src.sparse_voxel_gears.properties import SVProperties  # Import SVProperties for handling voxel properties
from src.sparse_voxel_gears.renderer import SVRenderer # render the model
from src.sparse_voxel_gears.adaptive import SVAdaptive # Handles adaptive voxel grid refinement, including subdivision and pruning
from src.sparse_voxel_gears.optimizer import SVOptimizer # Optimizer for the model
from src.sparse_voxel_gears.io import SVInOut # Handles input and output operations, including loading and saving models
from src.sparse_voxel_gears.pooling import SVPooling    # Handles pooling operations for the model


class SparseVoxelModel(SVConstructor, SVProperties, SVRenderer, SVAdaptive, SVOptimizer, SVInOut, SVPooling):

    def __init__(self, cfg_model):
        """
        Initializes the SparseVoxelModel with the given configuration.

        Args:
            cfg_model: Configuration object containing model parameters.

        Attributes:
            model_path (str): Path to the model file. # 모델 파일 경로
            vox_geo_mode (str): Mode for voxel geometry. # 복셀 기하 모드
            density_mode (str): Mode for density representation. # 밀도 표현 모드
            active_sh_degree (int): Active spherical harmonics degree. # 활성화된 구면 조화 함수 차수
            max_sh_degree (int): Maximum spherical harmonics degree. # 최대 구면 조화 함수 차수
            ss (bool): Whether supersampling is enabled. # 슈퍼샘플링 활성화 여부
            white_background (bool): Use white as the background color. # 흰색 배경 사용 여부
            black_background (bool): Use black as the background color. # 검은색 배경 사용 여부
            outside_level (int): Level for outside voxels, must be <= MAX_NUM_LEVELS. # 외부 복셀 레벨
            inside_level (int): Level for inside voxels, calculated as MAX_NUM_LEVELS - outside_level. # 내부 복셀 레벨
            per_voxel_attr_lst (list): List of per-voxel attributes. # 복셀별 속성 리스트
            per_voxel_param_lst (list): List of per-voxel parameters. # 복셀별 파라미터 리스트
            grid_pts_param_lst (list): List of grid points parameters. # 그리드 포인트 파라미터 리스트
            state_attr_names (list): Names of state attributes for optimization. # 최적화를 위한 상태 속성 이름 리스트
        """
        '''
        Setup of the model. The config is defined by `cfg.model` in `src/config.py`.
        After the initial setup. There are two ways to instantiate the models:

        1. `model_load` defined in `src/sparse_voxel_gears/io.py`.
           Load the saved models from a given path.

        2. `model_init` defined in `src/sparse_voxel_gears/constructor.py`.
           Heuristically initial the sparse grid layout and parameters from the training datas.
        '''
        super().__init__()
        self.model_path = cfg_model.model_path
        self.vox_geo_mode = cfg_model.vox_geo_mode
        self.density_mode = cfg_model.density_mode
        self.active_sh_degree = cfg_model.sh_degree
        self.max_sh_degree = cfg_model.sh_degree
        self.ss = cfg_model.ss
        self.white_background = cfg_model.white_background
        self.black_background = cfg_model.black_background

        assert cfg_model.outside_level <= svraster_cuda.meta.MAX_NUM_LEVELS
        self.outside_level = cfg_model.outside_level
        self.inside_level = svraster_cuda.meta.MAX_NUM_LEVELS - self.outside_level
        # List the variable names
        self.per_voxel_attr_lst = [
            'octpath',  # Octree path for the voxel
            'octlevel',  # Octree level of the voxel
            'vox_center',  # Center coordinates of the voxel
            'vox_size',  # Size of the voxel
            'subdiv_meta',  # Metadata for voxel subdivision (_subdiv_p 쭉 더한거 저장)
            'is_leaf',
        ]
        self.per_voxel_param_lst = [
            '_sh0',  # Spherical harmonics coefficient for degree 0
            '_shs',  # Spherical harmonics coefficients for higher degrees
            '_subdiv_p',  # Parameters for voxel subdivision (gradient tracking)
        ]
        self.grid_pts_param_lst = [
            '_geo_grid_pts',  # Geometric grid points for the voxel
        ]
        self.state_attr_names = [ #저장용
            'exp_avg',  # Exponential moving average for optimization
            'exp_avg_sq',  # Exponential moving average of squared gradients
        ]
