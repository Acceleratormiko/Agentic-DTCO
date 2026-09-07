#!/usr/bin/env python3
"""
训练DC和AC模型脚本
- DC模型：使用带导数的IV数据集（N型和P型）
- AC模型：使用CV数据集（N型和P型）
- 网络结构：参考RealMLP_EDA架构（7-16-32-8-1，GELU激活）
- 训练100个epoch
- 使用test_gx虚拟环境
"""

import gc
import builtins
import os
import re
import sys
import json
import random
import warnings

# 修复numpy兼容性问题
import numpy as np
# 为旧版pandas提供numpy兼容性（仅在需要时设置）
try:
    # 检查numpy版本
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    
    # 为旧版pandas提供兼容性别名（仅当不存在时）
    if not hasattr(np, 'bool'):
        try:
            np.bool = np.bool_
        except AttributeError:
            np.bool = bool
    
    if not hasattr(np, 'int'):
        try:
            np.int = np.int_
        except AttributeError:
            np.int = int
    
    if not hasattr(np, 'float'):
        try:
            np.float = np.float_
        except AttributeError:
            try:
                np.float = np.float64
            except AttributeError:
                np.float = float
except Exception:
    # 如果设置失败，继续执行（可能不需要这些别名）
    pass

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

warnings.filterwarnings('ignore')

# CompactModelTool可以只把本模块产生的训练输出写入独立日志，
# 不修改进程级sys.stdout/sys.stderr，避免影响Agent CLI自己的动态界面。
_TRAINING_OUTPUT_STREAM = None


def set_training_output(stream=None):
    """设置本模块print的目标流；传入None恢复普通终端输出。"""
    global _TRAINING_OUTPUT_STREAM
    _TRAINING_OUTPUT_STREAM = stream


def print(*args, **kwargs):
    """模块局部print代理，保持脚本单独运行时的原有行为。"""
    if _TRAINING_OUTPUT_STREAM is not None and "file" not in kwargs:
        kwargs["file"] = _TRAINING_OUTPUT_STREAM
    builtins.print(*args, **kwargs)

# ============================================================================
# IV/CV 预处理函数
# ============================================================================

def _ensure_vgs_vds(df):
    """确保数据集中存在 vgs/vds 列。"""
    df = df.copy()
    if 'vgs' not in df.columns:
        if 'vg' in df.columns and 'vs' in df.columns:
            df['vgs'] = df['vg'] - df['vs']
        else:
            raise ValueError("需要 vg, vs 列来构造 vgs")
    if 'vds' not in df.columns:
        if 'vd' in df.columns and 'vs' in df.columns:
            df['vds'] = df['vd'] - df['vs']
        else:
            raise ValueError("需要 vd, vs 列来构造 vds")
    return df


def interpolate_iv_to_csv(csv_path, out_path, vgs_factor=2, vds_factor=5):
    """
    对原始 IV 数据做 2x5 网格插值，供后续导数计算使用。
    """
    df = pd.read_csv(csv_path)
    df = _ensure_vgs_vds(df)
    row0 = df.iloc[0]
    l_val = row0['l'] if 'l' in df.columns else 0.018
    nfin_val = row0['nfin'] if 'nfin' in df.columns else 2
    eot_val = row0['eot'] if 'eot' in df.columns else 1.13e-9
    vs_val = row0['vs'] if 'vs' in df.columns else 0.0
    vb_val = row0['vb'] if 'vb' in df.columns else 0.0

    vgs = np.asarray(df['vgs'], dtype=float)
    vds = np.asarray(df['vds'], dtype=float)
    id_ = np.asarray(df['id'], dtype=float)
    points = np.column_stack([vgs, vds])

    u_vgs = np.unique(np.round(vgs, 10))
    u_vds = np.unique(np.round(vds, 10))
    step_vgs = (u_vgs.max() - u_vgs.min()) / (len(u_vgs) - 1) / vgs_factor if len(u_vgs) >= 2 else 0.01
    step_vds = (u_vds.max() - u_vds.min()) / (len(u_vds) - 1) / vds_factor if len(u_vds) >= 2 else 0.01
    new_vgs = np.unique(np.round(np.arange(vgs.min(), vgs.max() + step_vgs * 0.5, step_vgs), 10))
    new_vds = np.unique(np.round(np.arange(vds.min(), vds.max() + step_vds * 0.5, step_vds), 10))

    grid_vgs, grid_vds = np.meshgrid(new_vgs, new_vds, indexing='ij')
    grid_points = np.column_stack([grid_vgs.ravel(), grid_vds.ravel()])
    id_interp = griddata(points, id_, grid_points, method='linear', fill_value=np.nan)

    out = pd.DataFrame({
        'l': l_val,
        'nfin': nfin_val,
        'eot': eot_val,
        'vg': grid_vgs.ravel() + vs_val,
        'vd': grid_vds.ravel() + vs_val,
        'vs': vs_val,
        'vb': vb_val,
        'id': id_interp.ravel(),
        'vgs': grid_vgs.ravel(),
        'vds': grid_vds.ravel(),
    }).dropna(subset=['id'])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def cv_c_abs_to_path(csv_path, out_path, c_cols=None):
    """将 CV 数据中的电容列取绝对值后写出。"""
    c_cols = c_cols or ['cgg', 'cgs', 'cgd', 'cbg']
    df = pd.read_csv(csv_path)
    for col in c_cols:
        if col in df.columns:
            df[col] = df[col].abs()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

# ============================================================================
# IV数据集导数计算相关函数（从add_derivatives_to_iv.py集成）
# ============================================================================

def calculate_gm(vgs_array, id_array):
    """
    Calculate gm = dId/dVgs using numerical differentiation
    
    Args:
        vgs_array: Array of Vgs values (sorted)
        id_array: Array of Id values (corresponding to sorted Vgs)
        
    Returns:
        gm_array: Array of gm values
    """
    vgs_array = np.array(vgs_array)
    id_array = np.array(id_array)
    gm_array = np.zeros_like(id_array)
    
    if len(vgs_array) < 2:
        return gm_array
    
    # Calculate dId/dVgs using central difference
    for i in range(len(vgs_array)):
        if i == 0:
            # Forward difference
            if len(vgs_array) > 1:
                gm_array[i] = (id_array[i+1] - id_array[i]) / (vgs_array[i+1] - vgs_array[i])
            else:
                gm_array[i] = 0.0
        elif i == len(vgs_array) - 1:
            # Backward difference
            gm_array[i] = (id_array[i] - id_array[i-1]) / (vgs_array[i] - vgs_array[i-1])
        else:
            # Central difference
            gm_array[i] = (id_array[i+1] - id_array[i-1]) / (vgs_array[i+1] - vgs_array[i-1])
    
    return gm_array

def calculate_gds_7nm_style(device_df):
    """
    Calculate gds = dId/dVds (at fixed Vgs) using numerical differentiation
    参考7nm数据集的计算方式：固定Vgs，对Vds进行数值求导
    
    Args:
        device_df: DataFrame for a single device group, containing vgs, vds, id columns
        
    Returns:
        gds_dict: Dictionary mapping original index to gds value
    """
    device_df = device_df.copy()
    # 保存原始索引映射：排序后的位置 -> 原始索引
    device_df_sorted = device_df.sort_values(['vgs', 'vds'])
    index_mapping = device_df_sorted.index.values  # 排序后的原始索引顺序
    device_df_sorted = device_df_sorted.reset_index(drop=True)
    
    vgs_array = device_df_sorted['vgs'].values
    vds_array = device_df_sorted['vds'].values
    # Use original id values (negative for P-type, positive for N-type)
    id_array = device_df_sorted['id'].values
    
    # 使用字典存储gds值，键为原始索引
    gds_dict = {}
    
    # Group by Vgs and calculate gds for each Vgs group
    unique_vgs = np.unique(vgs_array)
    
    for vgs_val in unique_vgs:
        vgs_mask = np.abs(vgs_array - vgs_val) < 1e-6
        vgs_subset_indices = np.where(vgs_mask)[0]
        
        if len(vgs_subset_indices) < 2:
            continue
        
        # Get Vds and Id for this Vgs
        vds_subset = vds_array[vgs_subset_indices]
        id_subset = id_array[vgs_subset_indices]
        
        # Sort by Vds
        sort_idx = np.argsort(vds_subset)
        vds_sorted = vds_subset[sort_idx]
        id_sorted = id_subset[sort_idx]
        orig_indices = vgs_subset_indices[sort_idx]  # 排序后DataFrame的索引（位置）
        
        # Calculate gds using numerical differentiation
        for i, orig_idx in enumerate(orig_indices):
            # orig_idx是排序后DataFrame的位置索引（0,1,2...）
            # 需要映射回原始索引
            original_idx = index_mapping[orig_idx]
            
            if i == 0:
                # Forward difference
                if len(vds_sorted) > 1:
                    gds_val = (id_sorted[i+1] - id_sorted[i]) / (vds_sorted[i+1] - vds_sorted[i])
                    gds_dict[original_idx] = gds_val
                else:
                    gds_dict[original_idx] = 0.0
            elif i == len(vds_sorted) - 1:
                # Backward difference
                gds_val = (id_sorted[i] - id_sorted[i-1]) / (vds_sorted[i] - vds_sorted[i-1])
                gds_dict[original_idx] = gds_val
            else:
                # Central difference
                gds_val = (id_sorted[i+1] - id_sorted[i-1]) / (vds_sorted[i+1] - vds_sorted[i-1])
                gds_dict[original_idx] = gds_val
    
    return gds_dict

def add_derivatives_to_iv_csv(input_file, output_file=None):
    """
    为IV数据集CSV文件添加导数（gm和gds）
    处理IV数据的CSV文件，生成带导数的CSV文件
    
    Args:
        input_file: 输入的IV数据集CSV文件路径
        output_file: 输出的带导数CSV文件路径，如果为None则自动生成（在原文件名后加_with_gm_gds）
    
    Returns:
        str: 输出文件路径
    """
    import sys
    print(f"\n处理IV数据集添加导数: {input_file}")
    sys.stdout.flush()
    
    # 如果没有指定输出文件，自动生成
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_with_gm_gds{ext}"
    
    # Read input CSV with progress indication
    print("  读取CSV文件...")
    sys.stdout.flush()
    df = pd.read_csv(input_file)
    print(f"  读取了 {len(df)} 行")
    sys.stdout.flush()
    
    # For IV data, we have vg and vd, need to create vgs and vds
    # Assuming vs=0 for IV measurements
    if 'vgs' not in df.columns:
        if 'vg' in df.columns:
            df['vgs'] = df['vg']
        else:
            raise ValueError("Neither 'vgs' nor 'vg' column found")
    
    if 'vds' not in df.columns:
        if 'vd' in df.columns:
            df['vds'] = df['vd']
        else:
            raise ValueError("Neither 'vds' nor 'vd' column found")
    
    # Filter invalid data - use absolute value for filtering, but keep original for calculation
    df['id_abs'] = np.abs(df['id'])
    df = df[df['id_abs'] > 1e-20]  # Filter very small currents
    df = df.dropna(subset=['id', 'vgs', 'vds'])
    
    print(f"  过滤后: {len(df)} 行")
    
    # Device parameter columns (grouping key) - only l, nfin, eot for IV data
    device_cols = ['l', 'nfin', 'eot']
    
    # Group by device parameters
    print("  按设备参数分组...")
    sys.stdout.flush()
    device_groups = df.groupby(device_cols)
    # Convert to list to get count without consuming iterator
    device_groups_list = list(device_groups)
    num_groups = len(device_groups_list)
    print(f"  找到 {num_groups} 个设备组")
    sys.stdout.flush()
    
    # Initialize gm and gds columns
    df['gm'] = 0.0
    df['gds'] = 0.0
    
    # Calculate gm and gds for each device group
    print("  开始计算gm和gds...")
    sys.stdout.flush()
    # 不使用 tqdm 动态进度条。该模块运行在带动态界面的 Agent CLI 内，
    # tqdm 的回车刷新会与 CLI 重绘争用同一个终端。
    for idx, (device_key, device_df) in enumerate(device_groups_list):
        if idx % 100 == 0 and idx > 0:
            print(f"    已处理 {idx}/{num_groups} 个设备组")
            sys.stdout.flush()
        device_indices = device_df.index
        
        # Calculate gm: fixed Vds, vary Vgs
        # For P-type devices, keep original negative values (don't use absolute)
        for vds_val in device_df['vds'].unique():
            vds_mask = np.abs(device_df['vds'] - vds_val) < 1e-6
            vds_subset = device_df[vds_mask].copy()
            
            if len(vds_subset) < 2:
                continue
            
            # Sort by Vgs
            vds_subset = vds_subset.sort_values('vgs')
            vgs_vals = vds_subset['vgs'].values
            # Use original id values (negative for P-type, positive for N-type)
            id_vals = vds_subset['id'].values
            subset_indices = vds_subset.index.values
            
            # Calculate gm using original id values
            gm_vals = calculate_gm(vgs_vals, id_vals)
            
            # Update gm values
            for idx, gm_val in zip(subset_indices, gm_vals):
                df.loc[idx, 'gm'] = gm_val
        
        # Calculate gds: fixed Vgs, vary Vds (参考7nm数据集的计算方式)
        gds_dict = calculate_gds_7nm_style(device_df)
        for idx, gds_val in gds_dict.items():
            df.loc[idx, 'gds'] = gds_val
    
    # Filter out invalid gm and gds values
    df = df[np.isfinite(df['gm'])]
    df = df[np.isfinite(df['gds'])]
    
    print(f"  过滤无效导数后: {len(df)} 行")
    sys.stdout.flush()
    
    # Remove temporary column
    if 'id_abs' in df.columns:
        df = df.drop(columns=['id_abs'])
    
    # Save to output file
    print(f"  保存到: {output_file}")
    sys.stdout.flush()
    df.to_csv(output_file, index=False)
    print(f"  已保存到: {output_file}")
    sys.stdout.flush()
    
    return output_file


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """计算MAPE (Mean Absolute Percentage Error)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.maximum(np.abs(y_true), epsilon)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    return mape


def set_random_seeds(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_random_seeds(42)


class RealMLP_EDA(nn.Module):
    """
    RealMLP-EDA架构：参考训练脚本的网络结构
    架构: input(7) - 16 - 32 - 8 - output(3) [Id, gm, gds]
    激活函数: GELU
    初始化: Kaiming uniform (He初始化)
    """
    def __init__(self, input_size=7, output_size=3, dropout=0.1):
        super(RealMLP_EDA, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, output_size)
        
        self.activation = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化：Kaiming uniform (He初始化)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播，输出 [Id, gm, gds]"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x


class WeightedRMSLoss(nn.Module):
    """
    加权RMS损失函数
    直接对 Id, gm, gds 三个目标计算加权RMS和
    Id 是主要项
    """
    def __init__(self, weight_id=1.0, weight_gm=0.1, weight_gds=0.1, eps=1e-8):
        super(WeightedRMSLoss, self).__init__()
        self.weight_id = weight_id   # Id权重（主要项）
        self.weight_gm = weight_gm   # gm权重
        self.weight_gds = weight_gds  # gds权重
        self.eps = eps

    def forward(self, pred, target):
        """
        计算加权RMS损失
        
        Args:
            pred: 预测值，shape (N, 3) [Id, gm, gds]
            target: 真实值，shape (N, 3) [Id, gm, gds]
        
        Returns:
            total_loss, L_id, L_gm, L_gds
        """
        # 计算RMSE (Root Mean Squared Error)
        # RMSE = sqrt(mean((pred - true)^2))
        
        # Id的RMSE（主要项）
        mse_id = torch.mean((pred[:, 0:1] - target[:, 0:1]) ** 2)
        rmse_id = torch.sqrt(mse_id + self.eps)
        L_id = self.weight_id * rmse_id
        
        # gm的RMSE
        mse_gm = torch.mean((pred[:, 1:2] - target[:, 1:2]) ** 2)
        rmse_gm = torch.sqrt(mse_gm + self.eps)
        L_gm = self.weight_gm * rmse_gm
        
        # gds的RMSE
        mse_gds = torch.mean((pred[:, 2:3] - target[:, 2:3]) ** 2)
        rmse_gds = torch.sqrt(mse_gds + self.eps)
        L_gds = self.weight_gds * rmse_gds
        
        # 总损失
        L_total = L_id + L_gm + L_gds
        
        return L_total, L_id, L_gm, L_gds


def load_dc_data(csv_path, device_type='n'):
    """
    加载DC（IV）数据集
    严格按照train_bsim_iv_model.py的数据处理逻辑
    按l, nfin, eot分组处理，并生成node_key用于数据划分
    """
    print(f"\n加载DC数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  原始数据: {len(df)} 行")
    
    # 确保有必要的列
    if 'vs' not in df.columns:
        df['vs'] = 0.0
    if 'vb' not in df.columns:
        df['vb'] = 0.0
    
    if 'vgs' not in df.columns:
        if 'vg' in df.columns:
            df['vgs'] = df['vg'] - df['vs']
        else:
            raise ValueError("需要vg或vgs列")
    
    if 'vds' not in df.columns:
        if 'vd' in df.columns:
            df['vds'] = df['vd'] - df['vs']
        else:
            raise ValueError("需要vd或vds列")
    
    # 过滤无效数据（参考train_bsim_iv_model.py）
    if device_type == 'p':
        if 'id' in df.columns:
            # PMOS: 电流应该是正数（已经转换过）
            df = df[df['id'].abs() > 1e-20].copy()
    else:
        if 'id' in df.columns:
            df = df[df['id'].abs() > 1e-20].copy()
    
    print(f"  过滤id后: {len(df)} 行")
    
    # 过滤掉 |Vds|<1e-6 的点（避免 ln(Id/Vds) 数值异常）
    if 'vds' in df.columns and len(df) > 0:
        small = df['vds'].abs() < 1e-6
        n_small = int(small.sum())
        if n_small:
            df = df.loc[~small].copy()
        print(f"  |Vds|<1e-6 已过滤：{n_small} 行，剩余: {len(df)} 行")
    else:
        print(f"  |Vds|<1e-6 过滤: 0 行")
    
    # 检查必需列
    required_cols = ['l', 'nfin', 'eot', 'vg', 'vd', 'vs', 'vb', 'id', 'vds']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少列 {missing_cols}")
    
    # 填充NaN
    if len(df) > 0:
        if 'vs' in df.columns:
            df['vs'] = df['vs'].fillna(0.0)
        if 'vb' in df.columns:
            df['vb'] = df['vb'].fillna(0.0)
        if 'vd' in df.columns:
            df['vd'] = df['vd'].fillna(df['vds'] if 'vds' in df.columns else 0.0)
        # 只对关键列进行dropna
        critical_cols = ['l', 'nfin', 'eot', 'vg', 'id', 'vds', 'gm', 'gds', 'y']
        critical_cols = [col for col in critical_cols if col in df.columns]
        # 手动过滤NaN和inf
        mask = pd.Series([True] * len(df))
        for col in critical_cols:
            if col in df.columns:
                mask = mask & df[col].notna() & np.isfinite(df[col])
        df = df[mask].copy()
    
    print(f"  过滤NaN和inf后: {len(df)} 行")
    
    # 生成node_key（参考train_bsim_iv_model.py）
    # node_key基于(l, nfin, eot, vds)组合，每个唯一的组合对应一个node
    if 'node_key' not in df.columns:
        print("  根据几何参数(l, nfin, eot)和Vds值自动生成node_key")
        df['node_key'] = df.apply(
            lambda row: f"l_{row['l']:.6e}_nfin_{row['nfin']:.0f}_eot_{row['eot']:.6e}_vds_{row['vds']:.6e}",
            axis=1
        )
    
    print(f"  唯一device数量: {df['node_key'].nunique()}")
    print(f"  每个device平均点数: {len(df) / df['node_key'].nunique():.1f}")
    
    # 按l, nfin, eot分组处理（参考画图脚本的分组方式）
    device_cols = ['l', 'nfin', 'eot']
    device_groups = df.groupby(device_cols)
    print(f"  设备组数: {len(device_groups)}")
    
    # 合并所有组的数据
    all_data = []
    for group_key, group_df in device_groups:
        if len(group_df) > 0:
            all_data.append(group_df)
    
    if len(all_data) == 0:
        raise ValueError("没有有效的数据组")
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"  合并后: {len(df_combined)} 行")
    
    # 准备特征和目标
    feature_names = ['l', 'nfin', 'eot', 'vg', 'vd', 'vs', 'vb']
    X = df_combined[feature_names].values.astype(np.float32)
    
    # 输出目标：y=ln(Id/Vds), gm, gds（三个目标）；y 优先读 CSV 列，否则由 id/vds 现算
    gm_max = df_combined['gm'].abs().max()
    gds_max = df_combined['gds'].abs().max()

    if 'y' in df_combined.columns:
        y_y = df_combined['y'].values.astype(np.float64).reshape(-1, 1)
        print(f"  使用数据列 y=ln(Id/Vds)")
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            y_y = np.log(
                df_combined['id'].astype(np.float64).values
                / df_combined['vds'].astype(np.float64).values
            ).reshape(-1, 1)
        print(f"  未找到列 y，由 id/vds 计算 y=ln(Id/Vds)")
    
    if gm_max > 1.0:
        # 假设是mS单位，转换为S
        y_gm = (df_combined['gm'].values.astype(np.float32) / 1000.0).reshape(-1, 1)
        print(f"  检测到gm单位为mS，转换为S（最大值: {gm_max:.6e} mS）")
    else:
        # 已经是S单位
        y_gm = df_combined['gm'].values.astype(np.float32).reshape(-1, 1)
        print(f"  检测到gm单位为S（最大值: {gm_max:.6e} S）")
    
    if gds_max > 1.0:
        # 假设是mS单位，转换为S
        y_gds = (df_combined['gds'].values.astype(np.float32) / 1000.0).reshape(-1, 1)
        print(f"  检测到gds单位为mS，转换为S（最大值: {gds_max:.6e} mS）")
    else:
        # 已经是S单位
        y_gds = df_combined['gds'].values.astype(np.float32).reshape(-1, 1)
        print(f"  检测到gds单位为S（最大值: {gds_max:.6e} S）")
    
    # 合并三个目标
    y = np.hstack([y_y.astype(np.float32), y_gm, y_gds])  # Shape: (N, 3)
    
    vds_values = df_combined['vds'].values.reshape(-1, 1).astype(np.float32)
    vgs_values = df_combined['vgs'].values.reshape(-1, 1).astype(np.float32)
    node_keys = df_combined['node_key'].values
    
    print(f"  有效数据: X={X.shape}, y={y.shape}")
    if len(y) > 0:
        print(f"  y=ln(Id/Vds) 范围: [{y[:, 0].min():.6e}, {y[:, 0].max():.6e}]")
        print(f"  gm范围: [{y[:, 1].min():.6e}, {y[:, 1].max():.6e}] S")
        print(f"  gds范围: [{y[:, 2].min():.6e}, {y[:, 2].max():.6e}] S")
    
    return X, y, vds_values, vgs_values, node_keys, df_combined


def load_ac_data(csv_path, device_type='n'):
    """
    加载AC（CV）数据集，同时加载 cgs, cgd, cbtot, cgg 四个电容目标
    按l, nfin, eot分组处理
    """
    print(f"\n加载AC数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  原始数据: {len(df)} 行")
    
    # 确保有必要的列
    if 'vs' not in df.columns:
        df['vs'] = 0.0
    if 'vb' not in df.columns:
        df['vb'] = 0.0
    if 'vg' not in df.columns:
        if 'vgs' in df.columns:
            df['vg'] = df['vgs'] + df['vs']
        else:
            raise ValueError("需要vg或vgs列")
    if 'vd' not in df.columns:
        if 'vds' in df.columns:
            df['vd'] = df['vds'] + df['vs']
        else:
            raise ValueError("需要vd或vds列")
    
    # 目标列：cgs, cgd, cgg（不训练 cbtot）
    target_cols = ['cgs', 'cgd', 'cgg']
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"CV数据集缺少列: {c}")
    
    # 过滤无效数据：所有电容取绝对值后需大于阈值，并丢弃 NaN
    eps = 1e-20
    df = df.dropna(subset=['l', 'nfin', 'eot', 'vg', 'vd'] + target_cols)
    mask = np.ones(len(df), dtype=bool)
    for c in target_cols:
        mask &= (np.abs(df[c].values) > eps)
    df = df.loc[mask].copy()
    print(f"  过滤无效/缺失后: {len(df)} 行")
    
    # 按l, nfin, eot分组
    device_cols = ['l', 'nfin', 'eot']
    device_groups = df.groupby(device_cols)
    print(f"  设备组数: {len(device_groups)}")
    
    all_data = []
    for group_key, group_df in device_groups:
        if len(group_df) > 0:
            all_data.append(group_df)
    if len(all_data) == 0:
        raise ValueError("没有有效的数据组")
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"  合并后: {len(df_combined)} 行")
    
    feature_names = ['l', 'nfin', 'eot', 'vg', 'vd', 'vs', 'vb']
    X = df_combined[feature_names].values.astype(np.float32)
    # 三路输出：cgs, cgd, cgg
    y = df_combined[target_cols].values.astype(np.float32)
    print(f"  有效数据: X={X.shape}, y={y.shape} (cgs, cgd, cgg)")
    return X, y, df_combined


def train_dc_model(csv_path, output_dir, device_type='n', num_epochs=200):
    """训练DC模型"""
    print(f"\n{'='*60}")
    print(f"训练DC模型 ({device_type.upper()}-型)")
    print(f"{'='*60}")
    
    # 加载数据
    X, y, vds_values, vgs_values, node_keys, df_data = load_dc_data(csv_path, device_type)
    
    # 数据划分（严格按照train_bsim_iv_model.py的方式，按node_key分组，避免数据泄露）
    print("\n数据划分（按node_key分组，避免数据泄露）...")
    groups = node_keys
    indices = np.arange(len(X))
    
    # 使用GroupShuffleSplit确保同一device的所有点在同一数据集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    indices_temp, indices_test = next(gss.split(indices, groups=groups))
    
    # 对剩余数据再次分组划分
    groups_temp = groups[indices_temp]
    val_size = 0.15 / (0.7 + 0.15)  # val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    indices_train, indices_val = next(gss_val.split(indices_temp, groups=groups_temp))
    
    # 映射回原始索引
    indices_train = indices_temp[indices_train]
    indices_val = indices_temp[indices_val]
    
    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    y_train, y_val, y_test = y[indices_train], y[indices_val], y[indices_test]
    vds_train, vds_val, vds_test = vds_values[indices_train], vds_values[indices_val], vds_values[indices_test]
    vgs_train, vgs_val, vgs_test = vgs_values[indices_train], vgs_values[indices_val], vgs_values[indices_test]
    
    # 统计每个数据集的device数量
    train_node_keys = node_keys[indices_train]
    val_node_keys = node_keys[indices_val]
    test_node_keys = node_keys[indices_test]
    
    print(f"训练集: {len(X_train)} 点 ({len(np.unique(train_node_keys))} devices)")
    print(f"验证集: {len(X_val)} 点 ({len(np.unique(val_node_keys))} devices)")
    print(f"测试集: {len(X_test)} 点 ({len(np.unique(test_node_keys))} devices)")
    
    # 数据标准化（参考train_bsim_iv_model.py）
    scaler_X = StandardScaler()
    scaler_y_id = StandardScaler()
    scaler_y_gm = StandardScaler()
    scaler_y_gds = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 第一路为 y=ln(Id/Vds)，直接标准化；gm/gds 直接标准化
    iv_log_threshold = 1e-10
    y_train_id_scaled = scaler_y_id.fit_transform(y_train[:, 0:1])
    y_val_id_scaled = scaler_y_id.transform(y_val[:, 0:1])
    y_test_id_scaled = scaler_y_id.transform(y_test[:, 0:1])
    
    # gm和gds标准化（直接标准化）
    y_train_gm_scaled = scaler_y_gm.fit_transform(y_train[:, 1:2])
    y_val_gm_scaled = scaler_y_gm.transform(y_val[:, 1:2])
    y_test_gm_scaled = scaler_y_gm.transform(y_test[:, 1:2])
    
    y_train_gds_scaled = scaler_y_gds.fit_transform(y_train[:, 2:3])
    y_val_gds_scaled = scaler_y_gds.transform(y_val[:, 2:3])
    y_test_gds_scaled = scaler_y_gds.transform(y_test[:, 2:3])
    
    # 合并三个目标
    y_train_scaled = np.hstack([y_train_id_scaled, y_train_gm_scaled, y_train_gds_scaled])
    y_val_scaled = np.hstack([y_val_id_scaled, y_val_gm_scaled, y_val_gds_scaled])
    y_test_scaled = np.hstack([y_test_id_scaled, y_test_gm_scaled, y_test_gds_scaled])
    
    # 创建数据加载器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 将node_key转换为数字ID（用于后续可能的使用）
    le = LabelEncoder()
    all_node_keys = np.concatenate([train_node_keys, val_node_keys, test_node_keys])
    le.fit(all_node_keys)
    train_node_ids_encoded = le.transform(train_node_keys)
    val_node_ids_encoded = le.transform(val_node_keys)
    test_node_ids_encoded = le.transform(test_node_keys)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.LongTensor(train_node_ids_encoded),
        torch.FloatTensor(vgs_train),
        torch.FloatTensor(vds_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.LongTensor(val_node_ids_encoded),
        torch.FloatTensor(vgs_val),
        torch.FloatTensor(vds_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test_scaled),
        torch.LongTensor(test_node_ids_encoded),
        torch.FloatTensor(vgs_test),
        torch.FloatTensor(vds_test)
    )
    
    # 小数据集：batch_size 过大会导致每 epoch 梯度更新过少，难以达到 R²>0.995
    batch_size = min(16, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型（输出3个值：y, gm, gds）
    model = RealMLP_EDA(input_size=7, output_size=3).to(device)
    
    # 损失函数：三路输出均重要，权重平衡以提升 Id/gm/gds 的 R²
    criterion = WeightedRMSLoss(
        weight_id=1.0,
        weight_gm=0.5,
        weight_gds=0.5
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
    
    # 训练循环：200 epoch 内不提前停止，以充分收敛到 R²>0.995
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 200
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss_epoch = 0
        # batch 内部保持安静；每个 epoch 结束后只输出一条稳定日志。
        for batch_X, batch_y, batch_node_ids, batch_vgs, batch_vds in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred_scaled = model(batch_X)
            
            # 计算加权RMS损失
            loss, L_id, L_gm, L_gds = criterion(y_pred_scaled, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        # 验证
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_node_ids, batch_vgs, batch_vds in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                y_pred_scaled = model(batch_X)
                loss, _, _, _ = criterion(y_pred_scaled, batch_y)
                val_loss_epoch += loss.item()
        
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        
        scheduler.step(val_loss_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 每一轮都打印loss
        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: "
            f"Train Loss={train_loss_epoch:.6f}, "
            f"Val Loss={val_loss_epoch:.6f}, LR={current_lr:.6e}",
            flush=True,
        )
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 评估测试集
    model.eval()
    y_test_pred_scaled_list = []
    y_test_true_scaled_list = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_node_ids, batch_vgs, batch_vds in test_loader:
            batch_X = batch_X.to(device)
            y_pred_scaled = model(batch_X)
            y_test_pred_scaled_list.append(y_pred_scaled.cpu().numpy())
            y_test_true_scaled_list.append(batch_y.numpy())
    
    y_test_pred_scaled = np.vstack(y_test_pred_scaled_list)
    y_test_true_scaled = np.vstack(y_test_true_scaled_list)
    
    # 反标准化：第一路为 y=ln(Id/Vds)；gm和gds直接反标准化
    y_test_pred_id = scaler_y_id.inverse_transform(y_test_pred_scaled[:, 0:1])
    y_test_true_id = scaler_y_id.inverse_transform(y_test_true_scaled[:, 0:1])
    
    # gm和gds反标准化
    y_test_pred_gm = scaler_y_gm.inverse_transform(y_test_pred_scaled[:, 1:2])
    y_test_true_gm = scaler_y_gm.inverse_transform(y_test_true_scaled[:, 1:2])
    
    y_test_pred_gds = scaler_y_gds.inverse_transform(y_test_pred_scaled[:, 2:3])
    y_test_true_gds = scaler_y_gds.inverse_transform(y_test_true_scaled[:, 2:3])
    
    # 计算指标
    r2_id = r2_score(y_test_true_id, y_test_pred_id)
    mae_id = mean_absolute_error(y_test_true_id, y_test_pred_id)
    mape_id = calculate_mape(y_test_true_id, y_test_pred_id)
    
    r2_gm = r2_score(y_test_true_gm, y_test_pred_gm)
    mae_gm = mean_absolute_error(y_test_true_gm, y_test_pred_gm)
    mape_gm = calculate_mape(y_test_true_gm, y_test_pred_gm)
    
    r2_gds = r2_score(y_test_true_gds, y_test_pred_gds)
    mae_gds = mean_absolute_error(y_test_true_gds, y_test_pred_gds)
    mape_gds = calculate_mape(y_test_true_gds, y_test_pred_gds)
    
    print(f"\n{'='*60}")
    print(f"测试集指标 (DC模型 - {device_type.upper()}-型):")
    print(f"{'='*60}")
    print(f"y R²:   {r2_id:.6f}")
    print(f"gm R²:  {r2_gm:.6f}")
    print(f"gds R²: {r2_gds:.6f}")
    print(f"{'='*60}")
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"DC_model_{device_type}_type.pth"
    model_path = os.path.join(output_dir, model_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y_id': scaler_y_id,
        'scaler_y_gm': scaler_y_gm,
        'scaler_y_gds': scaler_y_gds,
        'iv_log_threshold': iv_log_threshold,
        'test_metrics': {
            'y': {'r2': float(r2_id), 'mae': float(mae_id), 'mape': float(mape_id)},
            'gm': {'r2': float(r2_gm), 'mae': float(mae_gm), 'mape': float(mape_gm)},
            'gds': {'r2': float(r2_gds), 'mae': float(mae_gds), 'mape': float(mape_gds)}
        }
    }, model_path)
    
    print(f"模型已保存到: {model_path}")
    
    # 显式释放模型和显存，避免多次调用时内存累积
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return None, scaler_X, scaler_y_id, scaler_y_gm, scaler_y_gds


def train_ac_model(csv_path, output_dir, device_type='n', num_epochs=200):
    """训练AC模型：同时拟合 cgs, cgd, cgg 并输出三路电容（不训练 cbtot）"""
    print(f"\n{'='*60}")
    print(f"训练AC模型 ({device_type.upper()}-型) [cgs, cgd, cgg]")
    print(f"{'='*60}")
    
    # 加载数据（y 为 Nx3: cgs, cgd, cgg）
    X, y, df_data = load_ac_data(csv_path, device_type)
    target_names = ['cgs', 'cgd', 'cgg']
    
    # 数据划分
    indices = np.arange(len(X))
    indices_temp, indices_test = train_test_split(indices, test_size=0.1, random_state=42)
    indices_train, indices_val = train_test_split(indices_temp, test_size=0.2, random_state=42)
    
    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    y_train, y_val, y_test = y[indices_train], y[indices_val], y[indices_test]
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 数据标准化（输入与四路输出）
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()
    
    X_train_scaled = scaler_input.fit_transform(X_train)
    X_val_scaled = scaler_input.transform(X_val)
    X_test_scaled = scaler_input.transform(X_test)
    
    y_train_scaled = scaler_output.fit_transform(y_train)
    y_val_scaled = scaler_output.transform(y_val)
    y_test_scaled = scaler_output.transform(y_test)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))
    
    batch_size = min(16, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 三路输出：cgs, cgd, cgg
    model = RealMLP_EDA(input_size=7, output_size=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 200
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        # batch 内部保持安静；每个 epoch 结束后只输出一条稳定日志。
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader)
        
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                y_pred = model(batch_X)
                val_loss_epoch += criterion(y_pred, batch_y).item()
        val_loss_epoch /= len(val_loader)
        scheduler.step()
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        print(
            f"Epoch {epoch+1:3d}/{num_epochs}: "
            f"Train Loss={train_loss_epoch:.6f}, Val Loss={val_loss_epoch:.6f}",
            flush=True,
        )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试集评估与输出
    model.eval()
    y_test_pred_list = []
    y_test_true_list = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            y_pred = model(batch_X)
            y_pred_orig = scaler_output.inverse_transform(y_pred.cpu().numpy())
            y_true_orig = scaler_output.inverse_transform(batch_y.cpu().numpy())
            y_test_pred_list.append(y_pred_orig)
            y_test_true_list.append(y_true_orig)
    
    y_test_pred = np.concatenate(y_test_pred_list)
    y_test_true = np.concatenate(y_test_true_list)
    
    # 各电容指标并输出
    test_metrics = {}
    print(f"\n{'='*60}")
    print(f"测试集指标 (AC模型 - {device_type.upper()}-型):")
    print(f"{'='*60}")
    for i, name in enumerate(target_names):
        r2 = r2_score(y_test_true[:, i], y_test_pred[:, i])
        mae = mean_absolute_error(y_test_true[:, i], y_test_pred[:, i])
        mape = calculate_mape(y_test_true[:, i], y_test_pred[:, i])
        test_metrics[name] = {'r2': float(r2), 'mae': float(mae), 'mape': float(mape)}
        print(f"{name}: R²={r2:.6f}, MAE={mae:.6e}, MAPE={mape:.2f}%")
    print(f"{'='*60}")
    
    # 将测试集预测与原数据对齐并写出带 cgs/cgd/cbtot/cgg 预测的 CSV（可选）
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"AC_CV_pred_{device_type}_type.csv")
    n_test = len(y_test_pred)
    out_df = pd.DataFrame({
        'cgs_true': y_test_true[:, 0],
        'cgs_pred': y_test_pred[:, 0],
        'cgd_true': y_test_true[:, 1],
        'cgd_pred': y_test_pred[:, 1],
        'cgg_true': y_test_true[:, 2],
        'cgg_pred': y_test_pred[:, 2],
    })
    out_df.to_csv(out_csv, index=False)
    print(f"测试集预测已输出: {out_csv}")
    
    model_name = f"AC_model_{device_type}_type.pth"
    model_path = os.path.join(output_dir, model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_input': scaler_input,
        'scaler_output': scaler_output,
        'output_names': target_names,
        'test_metrics': test_metrics,
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 显式释放模型和显存，避免多次调用时内存累积
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return None, scaler_input, scaler_output


def train_all_models(models_dir, n_iv_csv=None, p_iv_csv=None, n_cv_csv=None, p_cv_csv=None):
    """
    训练所有DC和AC模型（N型和P型）
    
    Args:
        models_dir: 模型保存目录，所有训练好的python model都保存在这里
        n_iv_csv: N型IV数据集CSV文件路径
        p_iv_csv: P型IV数据集CSV文件路径
        n_cv_csv: N型CV数据集CSV文件路径
        p_cv_csv: P型CV数据集CSV文件路径
    
    Returns:
        dict: 包含所有训练好的模型路径的字典
    """
    os.makedirs(models_dir, exist_ok=True)
    
    model_paths = {
        'dc_n': None,
        'dc_p': None,
        'ac_n': None,
        'ac_p': None
    }
    
    # 训练DC模型
    print("="*60)
    print("训练DC模型（Id, gm, gds）")
    print("="*60)
    
    # 处理N型IV数据集
    dc_n_csv = None
    if n_iv_csv and os.path.exists(n_iv_csv):
        # 检查是否已经有导数列
        try:
            df_check = pd.read_csv(n_iv_csv, nrows=1)
            if 'gm' not in df_check.columns or 'gds' not in df_check.columns:
                print(f"\n为N型IV数据集添加导数...")
                print(f"输入文件: {n_iv_csv}")
                try:
                    dc_n_csv = add_derivatives_to_iv_csv(n_iv_csv)
                    print(f"已生成带导数的N型IV数据集: {dc_n_csv}")
                except Exception as e:
                    print(f"警告: 为N型IV数据集添加导数失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果添加导数失败，尝试使用原始文件
                    dc_n_csv = n_iv_csv
            else:
                dc_n_csv = n_iv_csv
        except Exception as e:
            print(f"警告: 读取N型IV数据集失败: {e}")
            dc_n_csv = n_iv_csv
    elif n_iv_csv:
        print(f"警告: N型IV数据集文件不存在: {n_iv_csv}")
    
    # 处理P型IV数据集
    dc_p_csv = None
    if p_iv_csv and os.path.exists(p_iv_csv):
        # 检查是否已经有导数列
        try:
            df_check = pd.read_csv(p_iv_csv, nrows=1)
            if 'gm' not in df_check.columns or 'gds' not in df_check.columns:
                print(f"\n为P型IV数据集添加导数...")
                print(f"输入文件: {p_iv_csv}")
                try:
                    dc_p_csv = add_derivatives_to_iv_csv(p_iv_csv)
                    print(f"已生成带导数的P型IV数据集: {dc_p_csv}")
                except Exception as e:
                    print(f"警告: 为P型IV数据集添加导数失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果添加导数失败，尝试使用原始文件
                    dc_p_csv = p_iv_csv
            else:
                dc_p_csv = p_iv_csv
        except Exception as e:
            print(f"警告: 读取P型IV数据集失败: {e}")
            dc_p_csv = p_iv_csv
    elif p_iv_csv:
        print(f"警告: P型IV数据集文件不存在: {p_iv_csv}")
    
    if dc_n_csv:
        print(f"\n训练DC N型模型...")
        print(f"使用数据文件: {dc_n_csv}")
        train_dc_model(dc_n_csv, models_dir, device_type='n', num_epochs=200)
        model_paths['dc_n'] = os.path.join(models_dir, 'DC_model_n_type.pth')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"警告: 找不到DC N型数据文件")
    
    if dc_p_csv:
        print(f"\n训练DC P型模型...")
        print(f"使用数据文件: {dc_p_csv}")
        train_dc_model(dc_p_csv, models_dir, device_type='p', num_epochs=200)
        model_paths['dc_p'] = os.path.join(models_dir, 'DC_model_p_type.pth')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"警告: 找不到DC P型数据文件")
    
    # 训练AC模型
    print("\n" + "="*60)
    print("训练AC模型（cgs, cgd, cgg）")
    print("="*60)
    
    if n_cv_csv and os.path.exists(n_cv_csv):
        print(f"\n训练AC N型模型...")
        train_ac_model(n_cv_csv, models_dir, device_type='n', num_epochs=200)
        model_paths['ac_n'] = os.path.join(models_dir, 'AC_model_n_type.pth')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"警告: 找不到文件 {n_cv_csv}")
    
    if p_cv_csv and os.path.exists(p_cv_csv):
        print(f"\n训练AC P型模型...")
        train_ac_model(p_cv_csv, models_dir, device_type='p', num_epochs=200)
        model_paths['ac_p'] = os.path.join(models_dir, 'AC_model_p_type.pth')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"警告: 找不到文件 {p_cv_csv}")
    
    print("\n" + "="*60)
    print("所有模型训练完成！")
    print("="*60)

    return model_paths


# ============================================================================
# Verilog-A 转换相关函数和常量
# ============================================================================

# GELU 近似系数 (tanh 形式)
SQRT_2_OVER_PI = 0.7978845608028654   # sqrt(2/pi)
GELU_COEF = 0.044715

# VA 默认参数（与训练 scaler 一致，生成 .va 时写入 parameter）
DEFAULT_L_UM = 0.018   # um，与训练数据 l 量纲一致
DEFAULT_EOT_M = 1e-9   # m
DEFAULT_NFIN = 1


def _maybe_update_va_defaults_from_json(models_dir):
    """
    从导出的 params_dc_*.json 中回填 VA 默认几何参数。
    优先使用模型本身的 mean_X[:3]，避免 deck 侧再传 l/nfin/eot。
    """
    global DEFAULT_L_UM, DEFAULT_EOT_M, DEFAULT_NFIN
    for dtype in ['n', 'p']:
        dc_json = os.path.join(models_dir, f'params_dc_{dtype}.json')
        if not os.path.isfile(dc_json):
            continue
        try:
            with open(dc_json, 'r', encoding='utf-8') as f:
                dc_data = json.load(f)
            mean_x = dc_data.get('mean_X', [])
            if len(mean_x) < 3:
                continue
            DEFAULT_L_UM = float(mean_x[0])
            DEFAULT_NFIN = int(round(float(mean_x[1])))
            DEFAULT_EOT_M = float(mean_x[2])
            print(
                "VA defaults from model: "
                f"l={DEFAULT_L_UM} (um), nfin={DEFAULT_NFIN}, eot={DEFAULT_EOT_M} (m)"
            )
            return
        except Exception as exc:
            print(f"警告: 读取 {dc_json} 失败，继续使用现有 VA 默认参数: {exc}")


def _safe_scale(scale_arr):
    """仿 StandardScaler：scale 为 0 时置为 1 避免除零"""
    s = np.asarray(scale_arr, dtype=np.float64)
    s = np.where(np.abs(s) < 1e-30, 1.0, s)
    return s


def export_dc_model(pth_path, out_dir, device_type):
    """导出 DC (IV) 模型：权重、偏置、scaler 参数。"""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    scaler_X = ckpt['scaler_X']
    scaler_y_id = ckpt['scaler_y_id']
    scaler_y_gm = ckpt['scaler_y_gm']
    scaler_y_gds = ckpt['scaler_y_gds']
    iv_log_threshold = float(ckpt.get('iv_log_threshold', 1e-10))

    mean_X = np.array(scaler_X.mean_, dtype=np.float64).ravel()
    scale_X = _safe_scale(scaler_X.scale_)
    mean_id = np.array(scaler_y_id.mean_, dtype=np.float64).ravel()[0]
    scale_id = _safe_scale(scaler_y_id.scale_)[0]
    mean_gm = np.array(scaler_y_gm.mean_, dtype=np.float64).ravel()[0]
    scale_gm = _safe_scale(scaler_y_gm.scale_)[0]
    mean_gds = np.array(scaler_y_gds.mean_, dtype=np.float64).ravel()[0]
    scale_gds = _safe_scale(scaler_y_gds.scale_)[0]

    w1 = state['fc1.weight'].numpy()   # (16, 7)
    b1 = state['fc1.bias'].numpy()     # (16,)
    w2 = state['fc2.weight'].numpy()   # (32, 16)
    b2 = state['fc2.bias'].numpy()     # (32,)
    w3 = state['fc3.weight'].numpy()   # (8, 32)
    b3 = state['fc3.bias'].numpy()     # (8,)
    w4 = state['fc4.weight'].numpy()   # (3, 8)
    b4 = state['fc4.bias'].numpy()     # (3,)

    sign = -1.0 if device_type == 'p' else 1.0

    result = {
        'kind': 'dc',
        'device_type': device_type,
        'mean_X': mean_X.tolist(),
        'scale_X': scale_X.tolist(),
        'mean_id': mean_id,
        'scale_id': float(scale_id),
        'mean_gm': mean_gm,
        'scale_gm': float(scale_gm),
        'mean_gds': mean_gds,
        'scale_gds': float(scale_gds),
        'iv_log_threshold': iv_log_threshold,
        'sign': sign,
        'w1': w1.tolist(),
        'b1': b1.tolist(),
        'w2': w2.tolist(),
        'b2': b2.tolist(),
        'w3': w3.tolist(),
        'b3': b3.tolist(),
        'w4': w4.tolist(),
        'b4': b4.tolist(),
    }
    del ckpt
    gc.collect()
    return result


def export_ac_model(pth_path, out_dir, device_type):
    """导出 AC (CV) 模型：权重、偏置、scaler 参数。三路输出 [Cgs, Cgd, Cgg]，单位 F。"""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    scaler_input = ckpt['scaler_input']
    scaler_output = ckpt['scaler_output']

    mean_X = np.array(scaler_input.mean_, dtype=np.float64).ravel()
    scale_X = _safe_scale(scaler_input.scale_)
    mean_y = np.array(scaler_output.mean_, dtype=np.float64).ravel()   # (3,) for cgs, cgd, cgg
    scale_y = _safe_scale(scaler_output.scale_).ravel()
    if mean_y.size != 3:
        mean_y = np.atleast_1d(mean_y)
        scale_y = np.atleast_1d(scale_y)
        if mean_y.size == 1:
            mean_y = np.array([mean_y[0], mean_y[0], mean_y[0]], dtype=np.float64)
            scale_y = np.array([scale_y[0], scale_y[0], scale_y[0]], dtype=np.float64)
    # 保证 3 路：Cgs, Cgd, Cgg，单位与训练一致为 F
    mean_y = np.resize(mean_y, 3)
    scale_y = np.resize(scale_y, 3)

    w1 = state['fc1.weight'].numpy()
    b1 = state['fc1.bias'].numpy()
    w2 = state['fc2.weight'].numpy()
    b2 = state['fc2.bias'].numpy()
    w3 = state['fc3.weight'].numpy()
    b3 = state['fc3.bias'].numpy()
    w4 = state['fc4.weight'].numpy()   # (3, 8) for cgs, cgd, cgg
    b4 = state['fc4.bias'].numpy()    # (3,)
    if w4.ndim == 1:
        w4 = np.reshape(w4, (1, -1))
    if w4.shape[0] == 1:
        w4 = np.tile(w4, (3, 1))
        b4 = np.tile(np.atleast_1d(b4), 3)

    result = {
        'kind': 'ac',
        'device_type': device_type,
        'mean_X': mean_X.tolist(),
        'scale_X': scale_X.tolist(),
        'mean_y': mean_y.tolist(),
        'scale_y': scale_y.tolist(),
        'w1': w1.tolist(),
        'b1': b1.tolist(),
        'w2': w2.tolist(),
        'b2': b2.tolist(),
        'w3': w3.tolist(),
        'b3': b3.tolist(),
        'w4': w4.tolist(),
        'b4': b4.tolist(),
    }
    del ckpt
    gc.collect()
    return result


def _va_real(v):
    """格式化为 Verilog-A 实数常量；HSPICE PVA 不接受 1.e±xx，须为 1.0e±xx。"""
    x = float(v)
    if abs(x) < 1e-30:
        return '0.0'
    if abs(x - 1.0) < 1e-12:
        return '1.0'
    if abs(x + 1.0) < 1e-12:
        return '-1.0'
    # Special case: 1e-10 (common threshold value)
    if abs(x - 1e-10) < 1e-20:
        return '1.0e-10'
    s = np.format_float_scientific(x, precision=16, exp_digits=2)
    s = s.replace('E', 'e')
    s = re.sub(r'^(\d+)\.e([+\-])', r'\1.0e\2', s)
    s = re.sub(r'^-(\d+)\.e([+\-])', r'-\1.0e\2', s)
    if s.startswith('-1.e'):
        s = '-1.0e' + s[5:]
    elif s.startswith('1.e'):
        s = '1.0e' + s[4:]
    elif re.match(r'^-?1e[+\-]', s):
        s = re.sub(r'^(-?)(1)(e[+\-])', r'\g<1>1.0\g<3>', s)
    return s


def _gelu_va(var):
    """GELU 近似：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))。"""
    return f"0.5*({var})*(1 + tanh(0.7978845608*(({var}) + 0.044715*({var})*({var})*({var}))))"


def _generate_va_iv_full(data, out_path):
    """根据导出数据生成完整 IV 模块 Verilog-A（常量内联，可直接使用）。"""
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])
    lines = []
    lines.append("// Auto-generated IV (DC) Verilog-A module. Do not edit by hand.")
    lines.append("// Id 注入；gm/gds 由仿真器对 Id 求导得到；P-type 已恢复负号。")
    lines.append("")
    lines.append("module nnmodel_iv (g, d, s, b, gnd);")
    lines.append("  inout g, d, s, b, gnd;")
    lines.append("  electrical g, d, s, b, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    lines.append("  real xs0,xs1,xs2,xs3,xs4,xs5,xs6;")
    h1_names = [f"h1_{i}" for i in range(16)]
    h2_names = [f"h2_{i}" for i in range(32)]
    h3_names = [f"h3_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_names) + ";")
    lines.append("  real " + ", ".join(h2_names) + ";")
    lines.append("  real " + ", ".join(h3_names) + ";")
    lines.append("  real y0, y1, y2, y, Id_si, gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")
    for i in range(7):
        lines.append(f"    xs{i} = (x{i} - {_va_real(mean_X[i])}) / {_va_real(scale_X[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1[i, j])}*xs{j}" for j in range(7)]) + f" + {_va_real(b1[i])}"
        lines.append(f"    {h1_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_names[i])}; {h1_names[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2[i, j])}*{h1_names[j]}" for j in range(16)]) + f" + {_va_real(b2[i])}"
        lines.append(f"    {h2_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_names[i])}; {h2_names[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3[i, j])}*{h2_names[j]}" for j in range(32)]) + f" + {_va_real(b3[i])}"
        lines.append(f"    {h3_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_names[i])}; {h3_names[i]} = gi;")
    lines.append(f"    y0 = " + " + ".join([f"{_va_real(w4[0, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[0])};")
    lines.append(f"    y1 = " + " + ".join([f"{_va_real(w4[1, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[1])};")
    lines.append(f"    y2 = " + " + ".join([f"{_va_real(w4[2, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[2])};")
    lines.append(f"    y = y0 * {_va_real(data['scale_id'])} + {_va_real(data['mean_id'])};")
    lines.append("    Id_si = V(d,s) * exp(y);")
    lines.append("    I(d,s) <+ Id_si;")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _generate_va_cv_full(data, out_path):
    """根据导出数据生成完整 CV (AC) 模块 Verilog-A。三路输出 Cgs/Cgd/Cgg (F)，以 Cgs、Cgd 电荷形式注入。"""
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])
    mean_y = np.array(data['mean_y']).ravel()
    scale_y = np.array(data['scale_y']).ravel()
    mean_y = np.resize(mean_y, 3)
    scale_y = np.resize(scale_y, 3)
    lines = []
    lines.append("// Auto-generated CV (AC) Verilog-A. Cgs/Cgd from PTH (F); I(g,s)<+ ddt(Qgs), I(g,d)<+ ddt(Qgd).")
    lines.append("")
    lines.append("module nnmodel_cv (g, d, s, b, gnd);")
    lines.append("  inout g, d, s, b, gnd;")
    lines.append("  electrical g, d, s, b, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    lines.append("  real xs0,xs1,xs2,xs3,xs4,xs5,xs6;")
    h1_names = [f"h1_{i}" for i in range(16)]
    h2_names = [f"h2_{i}" for i in range(32)]
    h3_names = [f"h3_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_names) + ";")
    lines.append("  real " + ", ".join(h2_names) + ";")
    lines.append("  real " + ", ".join(h3_names) + ";")
    lines.append("  real y0, y1, y2, Cgs, Cgd, Cgg, Qgs, Qgd, gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")
    for i in range(7):
        lines.append(f"    xs{i} = (x{i} - {_va_real(mean_X[i])}) / {_va_real(scale_X[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1[i, j])}*xs{j}" for j in range(7)]) + f" + {_va_real(b1[i])}"
        lines.append(f"    {h1_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_names[i])}; {h1_names[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2[i, j])}*{h1_names[j]}" for j in range(16)]) + f" + {_va_real(b2[i])}"
        lines.append(f"    {h2_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_names[i])}; {h2_names[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3[i, j])}*{h2_names[j]}" for j in range(32)]) + f" + {_va_real(b3[i])}"
        lines.append(f"    {h3_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_names[i])}; {h3_names[i]} = gi;")
    lines.append(f"    y0 = " + " + ".join([f"{_va_real(w4[0, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[0])};")
    lines.append(f"    y1 = " + " + ".join([f"{_va_real(w4[1, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[1])};")
    lines.append(f"    y2 = " + " + ".join([f"{_va_real(w4[2, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[2])};")
    lines.append(f"    Cgs = y0 * {_va_real(scale_y[0])} + {_va_real(mean_y[0])};")
    lines.append(f"    Cgd = y1 * {_va_real(scale_y[1])} + {_va_real(mean_y[1])};")
    lines.append(f"    Cgg = y2 * {_va_real(scale_y[2])} + {_va_real(mean_y[2])};")
    lines.append("    if (Cgs < 1e-30) Cgs = 1e-30;")
    lines.append("    if (Cgd < 1e-30) Cgd = 1e-30;")
    lines.append("    Qgs = Cgs * V(g,s);")
    lines.append("    Qgd = Cgd * V(g,d);")
    lines.append("    I(g,s) <+ ddt(Qgs);")
    lines.append("    I(g,d) <+ ddt(Qgd);")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _generate_va_integrated(dc_data, ac_data, out_path, device_type):
    """生成 N 型或 P 型整合 VA：同一 module 内 IV + CV，供 HSPICE 仿真。"""
    # DC path
    mean_dc = np.array(dc_data['mean_X'])
    scale_dc = np.array(dc_data['scale_X'])
    w1_dc, b1_dc = np.array(dc_data['w1']), np.array(dc_data['b1'])
    w2_dc, b2_dc = np.array(dc_data['w2']), np.array(dc_data['b2'])
    w3_dc, b3_dc = np.array(dc_data['w3']), np.array(dc_data['b3'])
    w4_dc, b4_dc = np.array(dc_data['w4']), np.array(dc_data['b4'])
    # AC path
    mean_ac = np.array(ac_data['mean_X'])
    scale_ac = np.array(ac_data['scale_X'])
    w1_ac, b1_ac = np.array(ac_data['w1']), np.array(ac_data['b1'])
    w2_ac, b2_ac = np.array(ac_data['w2']), np.array(ac_data['b2'])
    w3_ac, b3_ac = np.array(ac_data['w3']), np.array(ac_data['b3'])
    w4_ac, b4_ac = np.array(ac_data['w4']), np.array(ac_data['b4'])

    mod_name = f"nnmodel_{device_type}_type"
    lines = []
    lines.append("// Auto-generated integrated Verilog-A: IV + CV, for HSPICE.")
    lines.append(f"// {device_type.upper()}-type: Id + Cgg in one module.")
    lines.append("")
    lines.append(f"module {mod_name} (g, d, s, b, cgs_out, cgd_out, cgg_out, gnd);")
    lines.append("  inout g, d, s, b, cgs_out, cgd_out, cgg_out, gnd;")
    lines.append("  electrical g, d, s, b, cgs_out, cgd_out, cgg_out, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  // [default off] junction diode: uncomment to enable")
    lines.append("  // parameter real Isj = 3e-5;   // junction saturation current")
    lines.append("  // parameter real Nj = 1.28;    // junction emission coefficient")
    lines.append("  // parameter real enable_junction = 0;  // 0 = off; 1 = normal with junction diode")
    if device_type == 'p':
        lines.append("  // [default off] gate_exp (PMOS off-region): uncomment to enable")
        lines.append("  // parameter real VTH_EFF  = 0.30;     // effective threshold (V)")
        lines.append("  // parameter real VTSCALE  = 0.02585;  // kT/q @ 300K (V)")
        lines.append("  // parameter real OFF_GAIN = 5.0;      // off-region slope")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    if device_type == 'n':
        lines.append("  // real Vbd, Vbs;  // junction diode (uncomment with junction params above)")
    else:
        lines.append("  // real gate_exp;  // PMOS off-region (uncomment with VTH_EFF above)")
        lines.append("  // real Vdb, Vsb;  // junction diode (uncomment with junction params above)")
    # DC
    lines.append("  real xs_dc_0,xs_dc_1,xs_dc_2,xs_dc_3,xs_dc_4,xs_dc_5,xs_dc_6;")
    h1_dc = [f"h1_dc_{i}" for i in range(16)]
    h2_dc = [f"h2_dc_{i}" for i in range(32)]
    h3_dc = [f"h3_dc_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_dc) + ";")
    lines.append("  real " + ", ".join(h2_dc) + ";")
    lines.append("  real " + ", ".join(h3_dc) + ";")
    lines.append("  real y0_dc, y, id_log, id_raw, Id_si;")
    # AC
    lines.append("  real xs_ac_0,xs_ac_1,xs_ac_2,xs_ac_3,xs_ac_4,xs_ac_5,xs_ac_6;")
    h1_ac = [f"h1_ac_{i}" for i in range(16)]
    h2_ac = [f"h2_ac_{i}" for i in range(32)]
    h3_ac = [f"h3_ac_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_ac) + ";")
    lines.append("  real " + ", ".join(h2_ac) + ";")
    lines.append("  real " + ", ".join(h3_ac) + ";")
    lines.append("  real y0_ac, y1_ac, y2_ac, Cgs, Cgd, Cgg;")
    lines.append("  real gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")

    # ---- DC path ----
    for i in range(7):
        lines.append(f"    xs_dc_{i} = (x{i} - {_va_real(mean_dc[i])}) / {_va_real(scale_dc[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1_dc[i, j])}*xs_dc_{j}" for j in range(7)]) + f" + {_va_real(b1_dc[i])}"
        lines.append(f"    {h1_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_dc[i])}; {h1_dc[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2_dc[i, j])}*{h1_dc[j]}" for j in range(16)]) + f" + {_va_real(b2_dc[i])}"
        lines.append(f"    {h2_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_dc[i])}; {h2_dc[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3_dc[i, j])}*{h2_dc[j]}" for j in range(32)]) + f" + {_va_real(b3_dc[i])}"
        lines.append(f"    {h3_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_dc[i])}; {h3_dc[i]} = gi;")
    lines.append(f"    y0_dc = " + " + ".join([f"{_va_real(w4_dc[0, j])}*{h3_dc[j]}" for j in range(8)]) + f" + {_va_real(b4_dc[0])};")
    _sid_dc = _va_real(dc_data['scale_id'])
    _mid_dc = _va_real(dc_data['mean_id'])
    lines.append(f"    y = y0_dc * {_sid_dc} + {_mid_dc};")
    if device_type == 'n':
        lines.append(f"    // id_log = y0_dc * {_sid_dc} + {_mid_dc};")
        lines.append("    // id_raw = exp(id_log) ;")
        lines.append("    //if (id_raw < 1e-30) id_raw = 1e-30;")
        lines.append("    // IV inject with optional tanh factor (原 Id 路径已弃用)")
        lines.append("    // if (enable_tanh_iv >= 0.5) begin")
        lines.append("    //   vds_eps = 0.02;")
        lines.append("    //   vds_eff = tanh(V(d,s)/vds_eps);")
        lines.append("    //   Id_si = id_raw * vds_eff;")
        lines.append("    // end else begin")
        lines.append("    //   Id_si = id_raw;")
        lines.append("    // end")
        lines.append("    // PMOS: vsd_eff = tanh(V(s,d)/vds_eps); Id_si = -id_raw * vsd_eff;")
    lines.append("    Id_si = V(d,s) * exp(y);")
    lines.append("    I(d,s) <+ Id_si;")
    if device_type == 'n':
        lines.append("    // [default off] NMOS junction diode: uncomment with junction params above")
        lines.append("    // if (enable_junction > 0.5) begin")
        lines.append("    //   Vbd = V(b,d); Vbs = V(b,s);")
        lines.append("    //   if (Vbd > 0) I(b,d) <+ Isj * (limexp(Vbd / (Nj * 0.02585)) - 1.0);")
        lines.append("    //   if (Vbs > 0) I(b,s) <+ Isj * (limexp(Vbs / (Nj * 0.02585)) - 1.0);")
        lines.append("    // end")
    else:
        lines.append("    // [default off] gate_exp: Id_si = -exp(y)*gate_exp; uncomment with VTH_EFF above")
        lines.append("    // gate_exp = exp((-V(g,s) - VTH_EFF) / (OFF_GAIN * VTSCALE));")
        lines.append("    // if (gate_exp > 1.0) gate_exp = 1.0;")
        lines.append("    // [default off] PMOS junction diode: uncomment with junction params above")
        lines.append("    // if (enable_junction > 0.5) begin")
        lines.append("    //   Vdb = V(d,b); Vsb = V(s,b);")
        lines.append("    //   if (Vdb > 0) I(d,b) <+ Isj * (limexp(Vdb / (Nj * 0.02585)) - 1.0);")
        lines.append("    //   if (Vsb > 0) I(s,b) <+ Isj * (limexp(Vsb / (Nj * 0.02585)) - 1.0);")
        lines.append("    // end")

    # ---- AC path ----
    for i in range(7):
        lines.append(f"    xs_ac_{i} = (x{i} - {_va_real(mean_ac[i])}) / {_va_real(scale_ac[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1_ac[i, j])}*xs_ac_{j}" for j in range(7)]) + f" + {_va_real(b1_ac[i])}"
        lines.append(f"    {h1_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_ac[i])}; {h1_ac[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2_ac[i, j])}*{h1_ac[j]}" for j in range(16)]) + f" + {_va_real(b2_ac[i])}"
        lines.append(f"    {h2_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_ac[i])}; {h2_ac[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3_ac[i, j])}*{h2_ac[j]}" for j in range(32)]) + f" + {_va_real(b3_ac[i])}"
        lines.append(f"    {h3_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_ac[i])}; {h3_ac[i]} = gi;")
    mean_ac_y = np.array(ac_data['mean_y']).ravel()
    scale_ac_y = np.array(ac_data['scale_y']).ravel()
    mean_ac_y = np.resize(mean_ac_y, 3)
    scale_ac_y = np.resize(scale_ac_y, 3)
    lines.append(f"    y0_ac = " + " + ".join([f"{_va_real(w4_ac[0, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[0])};")
    lines.append(f"    y1_ac = " + " + ".join([f"{_va_real(w4_ac[1, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[1])};")
    lines.append(f"    y2_ac = " + " + ".join([f"{_va_real(w4_ac[2, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[2])};")
    lines.append(f"    Cgs = y0_ac * {_va_real(scale_ac_y[0])} + {_va_real(mean_ac_y[0])};")
    lines.append(f"    Cgd = y1_ac * {_va_real(scale_ac_y[1])} + {_va_real(mean_ac_y[1])};")
    lines.append(f"    Cgg = y2_ac * {_va_real(scale_ac_y[2])} + {_va_real(mean_ac_y[2])};")
    lines.append("    if (Cgs < 1e-30) Cgs = 1e-30;")
    lines.append("    if (Cgd < 1e-30) Cgd = 1e-30;")
    if device_type == 'n':
        lines.append("    // numerical damping for TRAP (purely numerical, not physical)")
        lines.append("    I(g,s) <+ Cgs * ddt(V(g,s));")
        lines.append("    I(g,d) <+ Cgd * ddt(V(g,d));")
        lines.append("    I(g,s) <+ 1e-6 * V(g,s);")
        lines.append("    I(g,d) <+ 1e-6 * V(g,d);")
    else:
        lines.append("    // I = dQ/dt (charge-conserving); Q_gs = Cgs*V(g,s), Q_gd = Cgd*V(g,d)")
        lines.append("    I(g,s) <+ ddt(Cgs * V(g,s));")
        lines.append("    I(g,d) <+ ddt(Cgd * V(g,d));")
    lines.append("    // Output Cgs,Cgd,Cgg in F (SI) for HSPICE .print; plot script converts F->fF")
    lines.append("    V(cgs_out) <+ Cgs;")
    lines.append("    V(cgd_out) <+ Cgd;")
    lines.append("    V(cgg_out) <+ Cgg;")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _emit_params_inc(data, inc_path):
    """将导出数据写成 .inc 参数表（仅参数，供主 .va  include）。"""
    lines = []
    kind = data['kind']
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])

    lines.append('// Auto-generated parameter include')
    for i in range(7):
        lines.append(f'`define MEAN_X_{i} {_va_real(mean_X[i])}')
        lines.append(f'`define SCALE_X_{i} {_va_real(scale_X[i])}')
    for i in range(16):
        for j in range(7):
            lines.append(f'`define W1_{i}_{j} {_va_real(w1[i, j])}')
        lines.append(f'`define B1_{i} {_va_real(b1[i])}')
    for i in range(32):
        for j in range(16):
            lines.append(f'`define W2_{i}_{j} {_va_real(w2[i, j])}')
        lines.append(f'`define B2_{i} {_va_real(b2[i])}')
    for i in range(8):
        for j in range(32):
            lines.append(f'`define W3_{i}_{j} {_va_real(w3[i, j])}')
        lines.append(f'`define B3_{i} {_va_real(b3[i])}')
    if kind == 'dc':
        for j in range(8):
            lines.append(f'`define W4_0_{j} {_va_real(w4[0, j])}')
            lines.append(f'`define W4_1_{j} {_va_real(w4[1, j])}')
            lines.append(f'`define W4_2_{j} {_va_real(w4[2, j])}')
        lines.append(f'`define B4_0 {_va_real(b4[0])}')
        lines.append(f'`define B4_1 {_va_real(b4[1])}')
        lines.append(f'`define B4_2 {_va_real(b4[2])}')
        lines.append(f'`define MEAN_ID {_va_real(data["mean_id"])}')
        lines.append(f'`define SCALE_ID {_va_real(data["scale_id"])}')
        lines.append(f'`define MEAN_GM {_va_real(data["mean_gm"])}')
        lines.append(f'`define SCALE_GM {_va_real(data["scale_gm"])}')
        lines.append(f'`define MEAN_GDS {_va_real(data["mean_gds"])}')
        lines.append(f'`define SCALE_GDS {_va_real(data["scale_gds"])}')
        lines.append(f'`define IV_LOG_THRESH {_va_real(data["iv_log_threshold"])}')
        lines.append(f'`define SIGN_ID {_va_real(data["sign"])}')
    else:
        mean_y = np.array(data['mean_y']).ravel()
        scale_y = np.array(data['scale_y']).ravel()
        mean_y = np.resize(mean_y, 3)
        scale_y = np.resize(scale_y, 3)
        for j in range(8):
            lines.append(f'`define W4_0_{j} {_va_real(w4[0, j])}')
            lines.append(f'`define W4_1_{j} {_va_real(w4[1, j])}')
            lines.append(f'`define W4_2_{j} {_va_real(w4[2, j])}')
        lines.append(f'`define B4_0 {_va_real(b4[0])}')
        lines.append(f'`define B4_1 {_va_real(b4[1])}')
        lines.append(f'`define B4_2 {_va_real(b4[2])}')
        lines.append(f'`define MEAN_CGS {_va_real(mean_y[0])}')
        lines.append(f'`define MEAN_CGD {_va_real(mean_y[1])}')
        lines.append(f'`define MEAN_CGG {_va_real(mean_y[2])}')
        lines.append(f'`define SCALE_CGS {_va_real(scale_y[0])}')
        lines.append(f'`define SCALE_CGD {_va_real(scale_y[1])}')
        lines.append(f'`define SCALE_CGG {_va_real(scale_y[2])}')

    with open(inc_path, 'w') as f:
        f.write('\n'.join(lines))


def convert_models_to_va(models_dir):
    """
    将训练好的Python模型转换为Verilog-A模型
    
    Args:
        models_dir: 模型目录，包含pth文件和输出va文件
    
    Returns:
        list: 生成的va模型路径列表
    """
    print("\n" + "="*60)
    print("开始转换模型为Verilog-A格式")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    va_paths = []

    configs = [
        ('DC_model_n_type.pth', 'n', export_dc_model, 'params_dc_n'),
        ('DC_model_p_type.pth', 'p', export_dc_model, 'params_dc_p'),
        ('AC_model_n_type.pth', 'n', export_ac_model, 'params_ac_n'),
        ('AC_model_p_type.pth', 'p', export_ac_model, 'params_ac_p'),
    ]
    
    exported_data = {}
    
    # 导出所有模型
    for pth_name, dtype, exporter, base_name in configs:
        pth_path = os.path.join(models_dir, pth_name)
        if not os.path.isfile(pth_path):
            print(f"跳过（未找到）: {pth_path}")
            continue
        
        try:
            print(f"\n导出 {pth_name}...")
            data = exporter(pth_path, models_dir, dtype)
            json_path = os.path.join(models_dir, base_name + '.json')
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            inc_path = os.path.join(models_dir, base_name + '.inc')
            _emit_params_inc(data, inc_path)
            
            # 暂时注释掉生成单独的IV/CV VA模型，只生成整合的模型
            # if data['kind'] == 'dc':
            #     va_path = os.path.join(models_dir, f"nnmodel_iv_dc_{dtype}.va")
            #     _generate_va_iv_full(data, va_path)
            #     print(f"已导出: {pth_name} -> {json_path}, {inc_path}, {va_path}")
            # else:
            #     va_path = os.path.join(models_dir, f"nnmodel_cv_ac_{dtype}.va")
            #     _generate_va_cv_full(data, va_path)
            #     print(f"已导出: {pth_name} -> {json_path}, {inc_path}, {va_path}")
            print(f"已导出: {pth_name} -> {json_path}, {inc_path}")
            
            exported_data[base_name] = data
            
        except Exception as e:
            print(f"导出失败 {pth_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成整合的N型和P型VA模型
    print("\n" + "="*60)
    print("生成整合的Verilog-A模型")
    print("="*60)
    
    _maybe_update_va_defaults_from_json(models_dir)

    for dtype in ['n', 'p']:
        dc_json = os.path.join(models_dir, f'params_dc_{dtype}.json')
        ac_json = os.path.join(models_dir, f'params_ac_{dtype}.json')
        
        if os.path.isfile(dc_json) and os.path.isfile(ac_json):
            with open(dc_json) as f:
                dc_data = json.load(f)
            with open(ac_json) as f:
                ac_data = json.load(f)
            
            va_path = os.path.join(models_dir, f'nnmodel_{dtype}_type.va')
            _generate_va_integrated(dc_data, ac_data, va_path, dtype)
            va_paths.append(va_path)
            print(f"已生成整合 VA: {va_path} (IV+CV)")
        else:
            print(f"警告: 无法生成整合VA模型 {dtype} 型（缺少DC或AC模型）")
    
    print("\n" + "="*60)
    print("模型转换完成！")
    print("="*60)
    
    return va_paths


def main_block_with_preprocess(n_iv_csv=None, p_iv_csv=None, n_cv_csv=None, p_cv_csv=None, models_dir=None):
    """
    对 handover 风格数据做预处理后再进入训练/导出主流程。
    - CV: 电容列取绝对值
    - IV: 若缺少 gm/gds，则先在 vgs/vds 网格上做 2x5 插值
    """
    _default_models_dir = os.path.dirname(os.path.abspath(__file__))
    if models_dir is None:
        models_dir = _default_models_dir

    n_cv_use = n_cv_csv
    p_cv_use = p_cv_csv
    if n_cv_csv and os.path.exists(n_cv_csv):
        n_cv_use = os.path.join(models_dir, "n_cv_abs.csv")
        cv_c_abs_to_path(n_cv_csv, n_cv_use)
    if p_cv_csv and os.path.exists(p_cv_csv):
        p_cv_use = os.path.join(models_dir, "p_cv_abs.csv")
        cv_c_abs_to_path(p_cv_csv, p_cv_use)

    n_iv_use = n_iv_csv
    p_iv_use = p_iv_csv
    if n_iv_csv and os.path.exists(n_iv_csv):
        try:
            df_check = pd.read_csv(n_iv_csv, nrows=1)
            if "gm" not in df_check.columns or "gds" not in df_check.columns:
                n_iv_use = os.path.join(models_dir, "n_iv_interp_2x5.csv")
                interpolate_iv_to_csv(n_iv_csv, n_iv_use, vgs_factor=2, vds_factor=5)
        except Exception:
            n_iv_use = n_iv_csv
    if p_iv_csv and os.path.exists(p_iv_csv):
        try:
            df_check = pd.read_csv(p_iv_csv, nrows=1)
            if "gm" not in df_check.columns or "gds" not in df_check.columns:
                p_iv_use = os.path.join(models_dir, "p_iv_interp_2x5.csv")
                interpolate_iv_to_csv(p_iv_csv, p_iv_use, vgs_factor=2, vds_factor=5)
        except Exception:
            p_iv_use = p_iv_csv

    return main_block(
        n_iv_csv=n_iv_use,
        p_iv_csv=p_iv_use,
        n_cv_csv=n_cv_use,
        p_cv_csv=p_cv_use,
        models_dir=models_dir,
    )


def main_block(n_iv_csv=None, p_iv_csv=None, n_cv_csv=None, p_cv_csv=None, models_dir=None):
    """
    主函数：训练Python模型并自动转换为Verilog-A模型
    
    Args:
        n_iv_csv: N型IV数据集CSV文件路径
        p_iv_csv: P型IV数据集CSV文件路径
        n_cv_csv: N型CV数据集CSV文件路径
        p_cv_csv: P型CV数据集CSV文件路径
        models_dir: 模型保存目录，如果为None则使用默认路径
    """
    # 模型保存目录：由输入变量指定，默认使用 small_train 下的 test_tool
    _default_models_dir = os.path.dirname(os.path.abspath(__file__))
    if models_dir is None:
        models_dir = _default_models_dir
    
    print("="*60)
    print("开始训练和转换流程")
    print("="*60)
    print(f"模型保存目录: {models_dir}")
    print(f"N型IV数据集: {n_iv_csv}")
    print(f"P型IV数据集: {p_iv_csv}")
    print(f"N型CV数据集: {n_cv_csv}")
    print(f"P型CV数据集: {p_cv_csv}")
    print("="*60)
    
    # 步骤1: 训练所有Python模型（会自动为IV数据集添加导数）
    model_paths = train_all_models(models_dir, n_iv_csv, p_iv_csv, n_cv_csv, p_cv_csv)
    
    # 步骤2: 转换Python模型为Verilog-A模型
    va_paths = convert_models_to_va(models_dir)
    
    # 输出结果
    print("\n" + "="*60)
    print("流程完成！")
    print("="*60)
    print("\n训练好的Python模型:")
    for key, path in model_paths.items():
        if path and os.path.exists(path):
            print(f"  {key}: {path}")
    
    print("\n生成的Verilog-A模型:")
    for va_path in va_paths:
        if os.path.exists(va_path):
            print(f"  {va_path}")
    
    print("\n所有文件已保存到:", models_dir)
    
    # 返回生成的va模型路径（用于后续可能的使用）
    return va_paths


def dataset_to_va(n_iv_csv, p_iv_csv, n_cv_csv, p_cv_csv, models_dir=None):
    """
    封装好的函数接口：从四个CSV数据集训练模型并生成N/P型Verilog-A模型。
    
    该函数用于在其他模块（例如 main.py）中直接调用，而无需关心内部训练与转换细节。
    
    Args:
        n_iv_csv: N型IV数据集CSV文件路径
        p_iv_csv: P型IV数据集CSV文件路径
        n_cv_csv: N型CV数据集CSV文件路径
        p_cv_csv: P型CV数据集CSV文件路径
        models_dir: 模型和VA文件的保存目录，如果为None则使用当前脚本所在目录
    
    Returns:
        (p_va_path, n_va_path):
            p_va_path 为P型整合VA模型路径（nnmodel_p_type.va），
            n_va_path 为N型整合VA模型路径（nnmodel_n_type.va）。
    """
    # 直接复用本文件中的 main 流程
    va_paths = main_block_with_preprocess(
        n_iv_csv=n_iv_csv,
        p_iv_csv=p_iv_csv,
        n_cv_csv=n_cv_csv,
        p_cv_csv=p_cv_csv,
        models_dir=models_dir,
    )

    # 根据文件名识别 N / P 型 VA 路径，并按照 (p, n) 的顺序返回，方便后续电路评估模块使用
    n_va_path = None
    p_va_path = None
    for path in va_paths:
        base = os.path.basename(path) if path else ""
        if "nnmodel_n_type.va" in base:
            n_va_path = path
        elif "nnmodel_p_type.va" in base:
            p_va_path = path

    # 转换为绝对路径
    if p_va_path:
        p_va_path = os.path.abspath(p_va_path)
    if n_va_path:
        n_va_path = os.path.abspath(n_va_path)

    return p_va_path, n_va_path


# if __name__ == "__main__":
#     import argparse
#     _script_dir = os.path.dirname(os.path.abspath(__file__))
#     _default_models = _script_dir
#     parser = argparse.ArgumentParser(description='训练DC和AC模型并转换为Verilog-A格式')
#     parser.add_argument('--n_iv_csv', type=str, required=True,
#                         help='N型IV数据集CSV文件路径')
#     parser.add_argument('--p_iv_csv', type=str, required=True,
#                         help='P型IV数据集CSV文件路径')
#     parser.add_argument('--n_cv_csv', type=str, required=True,
#                         help='N型CV数据集CSV文件路径')
#     parser.add_argument('--p_cv_csv', type=str, required=True,
#                         help='P型CV数据集CSV文件路径')
#     parser.add_argument('--models_dir', type=str, default=None,
#                         help='模型保存目录路径（默认: test_tool）')
#     args = parser.parse_args()
#     main(n_iv_csv=args.n_iv_csv, p_iv_csv=args.p_iv_csv, 
#          n_cv_csv=args.n_cv_csv, p_cv_csv=args.p_cv_csv, 
#          models_dir=args.models_dir or _default_models)


def export_dc_model(pth_path, out_dir, device_type):
    """导出 DC (IV) 模型：权重、偏置、scaler 参数。"""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    scaler_X = ckpt['scaler_X']
    scaler_y_id = ckpt['scaler_y_id']
    scaler_y_gm = ckpt['scaler_y_gm']
    scaler_y_gds = ckpt['scaler_y_gds']
    iv_log_threshold = float(ckpt.get('iv_log_threshold', 1e-10))

    mean_X = np.array(scaler_X.mean_, dtype=np.float64).ravel()
    scale_X = _safe_scale(scaler_X.scale_)
    mean_id = np.array(scaler_y_id.mean_, dtype=np.float64).ravel()[0]
    scale_id = _safe_scale(scaler_y_id.scale_)[0]
    mean_gm = np.array(scaler_y_gm.mean_, dtype=np.float64).ravel()[0]
    scale_gm = _safe_scale(scaler_y_gm.scale_)[0]
    mean_gds = np.array(scaler_y_gds.mean_, dtype=np.float64).ravel()[0]
    scale_gds = _safe_scale(scaler_y_gds.scale_)[0]

    w1 = state['fc1.weight'].numpy()   # (16, 7)
    b1 = state['fc1.bias'].numpy()     # (16,)
    w2 = state['fc2.weight'].numpy()   # (32, 16)
    b2 = state['fc2.bias'].numpy()     # (32,)
    w3 = state['fc3.weight'].numpy()   # (8, 32)
    b3 = state['fc3.bias'].numpy()     # (8,)
    w4 = state['fc4.weight'].numpy()   # (3, 8)
    b4 = state['fc4.bias'].numpy()     # (3,)

    sign = -1.0 if device_type == 'p' else 1.0

    result = {
        'kind': 'dc',
        'device_type': device_type,
        'mean_X': mean_X.tolist(),
        'scale_X': scale_X.tolist(),
        'mean_id': mean_id,
        'scale_id': float(scale_id),
        'mean_gm': mean_gm,
        'scale_gm': float(scale_gm),
        'mean_gds': mean_gds,
        'scale_gds': float(scale_gds),
        'iv_log_threshold': iv_log_threshold,
        'sign': sign,
        'w1': w1.tolist(),
        'b1': b1.tolist(),
        'w2': w2.tolist(),
        'b2': b2.tolist(),
        'w3': w3.tolist(),
        'b3': b3.tolist(),
        'w4': w4.tolist(),
        'b4': b4.tolist(),
    }
    del ckpt
    gc.collect()
    return result


def export_ac_model(pth_path, out_dir, device_type):
    """导出 AC (CV) 模型：权重、偏置、scaler 参数。三路输出 [Cgs, Cgd, Cgg]，单位 F。"""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    scaler_input = ckpt['scaler_input']
    scaler_output = ckpt['scaler_output']

    mean_X = np.array(scaler_input.mean_, dtype=np.float64).ravel()
    scale_X = _safe_scale(scaler_input.scale_)
    mean_y = np.array(scaler_output.mean_, dtype=np.float64).ravel()   # (3,) for cgs, cgd, cgg
    scale_y = _safe_scale(scaler_output.scale_).ravel()
    if mean_y.size != 3:
        mean_y = np.atleast_1d(mean_y)
        scale_y = np.atleast_1d(scale_y)
        if mean_y.size == 1:
            mean_y = np.array([mean_y[0], mean_y[0], mean_y[0]], dtype=np.float64)
            scale_y = np.array([scale_y[0], scale_y[0], scale_y[0]], dtype=np.float64)
    # 保证 3 路：Cgs, Cgd, Cgg，单位与训练一致为 F
    mean_y = np.resize(mean_y, 3)
    scale_y = np.resize(scale_y, 3)

    w1 = state['fc1.weight'].numpy()
    b1 = state['fc1.bias'].numpy()
    w2 = state['fc2.weight'].numpy()
    b2 = state['fc2.bias'].numpy()
    w3 = state['fc3.weight'].numpy()
    b3 = state['fc3.bias'].numpy()
    w4 = state['fc4.weight'].numpy()   # (3, 8) for cgs, cgd, cgg
    b4 = state['fc4.bias'].numpy()    # (3,)
    if w4.ndim == 1:
        w4 = np.reshape(w4, (1, -1))
    if w4.shape[0] == 1:
        w4 = np.tile(w4, (3, 1))
        b4 = np.tile(np.atleast_1d(b4), 3)

    result = {
        'kind': 'ac',
        'device_type': device_type,
        'mean_X': mean_X.tolist(),
        'scale_X': scale_X.tolist(),
        'mean_y': mean_y.tolist(),
        'scale_y': scale_y.tolist(),
        'w1': w1.tolist(),
        'b1': b1.tolist(),
        'w2': w2.tolist(),
        'b2': b2.tolist(),
        'w3': w3.tolist(),
        'b3': b3.tolist(),
        'w4': w4.tolist(),
        'b4': b4.tolist(),
    }
    del ckpt
    gc.collect()
    return result


def _va_real(v):
    """格式化为 Verilog-A 实数常量；HSPICE PVA 不接受 1.e±xx，须为 1.0e±xx。"""
    x = float(v)
    if abs(x) < 1e-30:
        return '0.0'
    if abs(x - 1.0) < 1e-12:
        return '1.0'
    if abs(x + 1.0) < 1e-12:
        return '-1.0'
    # Special case: 1e-10 (common threshold value)
    if abs(x - 1e-10) < 1e-20:
        return '1.0e-10'
    s = np.format_float_scientific(x, precision=16, exp_digits=2)
    s = s.replace('E', 'e')
    s = re.sub(r'^(\d+)\.e([+\-])', r'\1.0e\2', s)
    s = re.sub(r'^-(\d+)\.e([+\-])', r'-\1.0e\2', s)
    if s.startswith('-1.e'):
        s = '-1.0e' + s[5:]
    elif s.startswith('1.e'):
        s = '1.0e' + s[4:]
    elif re.match(r'^-?1e[+\-]', s):
        s = re.sub(r'^(-?)(1)(e[+\-])', r'\g<1>1.0\g<3>', s)
    return s


def _gelu_va(var):
    """GELU 近似：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))。"""
    return f"0.5*({var})*(1 + tanh(0.7978845608*(({var}) + 0.044715*({var})*({var})*({var}))))"


def _generate_va_iv_full(data, out_path):
    """根据导出数据生成完整 IV 模块 Verilog-A（常量内联，可直接使用）。"""
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])
    lines = []
    lines.append("// Auto-generated IV (DC) Verilog-A module. Do not edit by hand.")
    lines.append("// Id 注入；gm/gds 由仿真器对 Id 求导得到；P-type 已恢复负号。")
    lines.append("")
    lines.append("module nnmodel_iv (g, d, s, b, gnd);")
    lines.append("  inout g, d, s, b, gnd;")
    lines.append("  electrical g, d, s, b, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    lines.append("  real xs0,xs1,xs2,xs3,xs4,xs5,xs6;")
    h1_names = [f"h1_{i}" for i in range(16)]
    h2_names = [f"h2_{i}" for i in range(32)]
    h3_names = [f"h3_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_names) + ";")
    lines.append("  real " + ", ".join(h2_names) + ";")
    lines.append("  real " + ", ".join(h3_names) + ";")
    lines.append("  real y0, y1, y2, y, Id_si, gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")
    for i in range(7):
        lines.append(f"    xs{i} = (x{i} - {_va_real(mean_X[i])}) / {_va_real(scale_X[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1[i, j])}*xs{j}" for j in range(7)]) + f" + {_va_real(b1[i])}"
        lines.append(f"    {h1_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_names[i])}; {h1_names[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2[i, j])}*{h1_names[j]}" for j in range(16)]) + f" + {_va_real(b2[i])}"
        lines.append(f"    {h2_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_names[i])}; {h2_names[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3[i, j])}*{h2_names[j]}" for j in range(32)]) + f" + {_va_real(b3[i])}"
        lines.append(f"    {h3_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_names[i])}; {h3_names[i]} = gi;")
    lines.append(f"    y0 = " + " + ".join([f"{_va_real(w4[0, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[0])};")
    lines.append(f"    y1 = " + " + ".join([f"{_va_real(w4[1, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[1])};")
    lines.append(f"    y2 = " + " + ".join([f"{_va_real(w4[2, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[2])};")
    lines.append(f"    y = y0 * {_va_real(data['scale_id'])} + {_va_real(data['mean_id'])};")
    lines.append("    Id_si = V(d,s) * exp(y);")
    lines.append("    I(d,s) <+ Id_si;")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _generate_va_cv_full(data, out_path):
    """根据导出数据生成完整 CV (AC) 模块 Verilog-A。三路输出 Cgs/Cgd/Cgg (F)，以 Cgs、Cgd 电荷形式注入。"""
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])
    mean_y = np.array(data['mean_y']).ravel()
    scale_y = np.array(data['scale_y']).ravel()
    mean_y = np.resize(mean_y, 3)
    scale_y = np.resize(scale_y, 3)
    lines = []
    lines.append("// Auto-generated CV (AC) Verilog-A. Cgs/Cgd from PTH (F); I(g,s)<+ ddt(Qgs), I(g,d)<+ ddt(Qgd).")
    lines.append("")
    lines.append("module nnmodel_cv (g, d, s, b, gnd);")
    lines.append("  inout g, d, s, b, gnd;")
    lines.append("  electrical g, d, s, b, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    lines.append("  real xs0,xs1,xs2,xs3,xs4,xs5,xs6;")
    h1_names = [f"h1_{i}" for i in range(16)]
    h2_names = [f"h2_{i}" for i in range(32)]
    h3_names = [f"h3_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_names) + ";")
    lines.append("  real " + ", ".join(h2_names) + ";")
    lines.append("  real " + ", ".join(h3_names) + ";")
    lines.append("  real y0, y1, y2, Cgs, Cgd, Cgg, Qgs, Qgd, gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")
    for i in range(7):
        lines.append(f"    xs{i} = (x{i} - {_va_real(mean_X[i])}) / {_va_real(scale_X[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1[i, j])}*xs{j}" for j in range(7)]) + f" + {_va_real(b1[i])}"
        lines.append(f"    {h1_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_names[i])}; {h1_names[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2[i, j])}*{h1_names[j]}" for j in range(16)]) + f" + {_va_real(b2[i])}"
        lines.append(f"    {h2_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_names[i])}; {h2_names[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3[i, j])}*{h2_names[j]}" for j in range(32)]) + f" + {_va_real(b3[i])}"
        lines.append(f"    {h3_names[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_names[i])}; {h3_names[i]} = gi;")
    lines.append(f"    y0 = " + " + ".join([f"{_va_real(w4[0, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[0])};")
    lines.append(f"    y1 = " + " + ".join([f"{_va_real(w4[1, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[1])};")
    lines.append(f"    y2 = " + " + ".join([f"{_va_real(w4[2, j])}*{h3_names[j]}" for j in range(8)]) + f" + {_va_real(b4[2])};")
    lines.append(f"    Cgs = y0 * {_va_real(scale_y[0])} + {_va_real(mean_y[0])};")
    lines.append(f"    Cgd = y1 * {_va_real(scale_y[1])} + {_va_real(mean_y[1])};")
    lines.append(f"    Cgg = y2 * {_va_real(scale_y[2])} + {_va_real(mean_y[2])};")
    lines.append("    if (Cgs < 1e-30) Cgs = 1e-30;")
    lines.append("    if (Cgd < 1e-30) Cgd = 1e-30;")
    lines.append("    Qgs = Cgs * V(g,s);")
    lines.append("    Qgd = Cgd * V(g,d);")
    lines.append("    I(g,s) <+ ddt(Qgs);")
    lines.append("    I(g,d) <+ ddt(Qgd);")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _generate_va_integrated(dc_data, ac_data, out_path, device_type):
    """生成 N 型或 P 型整合 VA：同一 module 内 IV + CV，供 HSPICE 仿真。"""
    # DC path
    mean_dc = np.array(dc_data['mean_X'])
    scale_dc = np.array(dc_data['scale_X'])
    w1_dc, b1_dc = np.array(dc_data['w1']), np.array(dc_data['b1'])
    w2_dc, b2_dc = np.array(dc_data['w2']), np.array(dc_data['b2'])
    w3_dc, b3_dc = np.array(dc_data['w3']), np.array(dc_data['b3'])
    w4_dc, b4_dc = np.array(dc_data['w4']), np.array(dc_data['b4'])
    # AC path
    mean_ac = np.array(ac_data['mean_X'])
    scale_ac = np.array(ac_data['scale_X'])
    w1_ac, b1_ac = np.array(ac_data['w1']), np.array(ac_data['b1'])
    w2_ac, b2_ac = np.array(ac_data['w2']), np.array(ac_data['b2'])
    w3_ac, b3_ac = np.array(ac_data['w3']), np.array(ac_data['b3'])
    w4_ac, b4_ac = np.array(ac_data['w4']), np.array(ac_data['b4'])

    mod_name = f"nnmodel_{device_type}_type"
    lines = []
    lines.append("// Auto-generated integrated Verilog-A: IV + CV, for HSPICE.")
    lines.append(f"// {device_type.upper()}-type: Id + Cgg in one module.")
    lines.append("")
    lines.append(f"module {mod_name} (g, d, s, b, cgs_out, cgd_out, cgg_out, gnd);")
    lines.append("  inout g, d, s, b, cgs_out, cgd_out, cgg_out, gnd;")
    lines.append("  electrical g, d, s, b, cgs_out, cgd_out, cgg_out, gnd;")
    lines.append(f"  parameter real l = {DEFAULT_L_UM};  // um (match training scaler); do NOT use meters")
    lines.append(f"  parameter real nfin = {DEFAULT_NFIN};")
    lines.append(f"  parameter real eot = {DEFAULT_EOT_M};  // m")
    lines.append("  // [default off] junction diode: uncomment to enable")
    lines.append("  // parameter real Isj = 3e-5;   // junction saturation current")
    lines.append("  // parameter real Nj = 1.28;    // junction emission coefficient")
    lines.append("  // parameter real enable_junction = 0;  // 0 = off; 1 = normal with junction diode")
    if device_type == 'p':
        lines.append("  // [default off] gate_exp (PMOS off-region): uncomment to enable")
        lines.append("  // parameter real VTH_EFF  = 0.30;     // effective threshold (V)")
        lines.append("  // parameter real VTSCALE  = 0.02585;  // kT/q @ 300K (V)")
        lines.append("  // parameter real OFF_GAIN = 5.0;      // off-region slope")
    lines.append("  real x0,x1,x2,x3,x4,x5,x6;")
    if device_type == 'n':
        lines.append("  // real Vbd, Vbs;  // junction diode (uncomment with junction params above)")
    else:
        lines.append("  // real gate_exp;  // PMOS off-region (uncomment with VTH_EFF above)")
        lines.append("  // real Vdb, Vsb;  // junction diode (uncomment with junction params above)")
    # DC
    lines.append("  real xs_dc_0,xs_dc_1,xs_dc_2,xs_dc_3,xs_dc_4,xs_dc_5,xs_dc_6;")
    h1_dc = [f"h1_dc_{i}" for i in range(16)]
    h2_dc = [f"h2_dc_{i}" for i in range(32)]
    h3_dc = [f"h3_dc_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_dc) + ";")
    lines.append("  real " + ", ".join(h2_dc) + ";")
    lines.append("  real " + ", ".join(h3_dc) + ";")
    lines.append("  real y0_dc, y, id_log, id_raw, Id_si;")
    # AC
    lines.append("  real xs_ac_0,xs_ac_1,xs_ac_2,xs_ac_3,xs_ac_4,xs_ac_5,xs_ac_6;")
    h1_ac = [f"h1_ac_{i}" for i in range(16)]
    h2_ac = [f"h2_ac_{i}" for i in range(32)]
    h3_ac = [f"h3_ac_{i}" for i in range(8)]
    lines.append("  real " + ", ".join(h1_ac) + ";")
    lines.append("  real " + ", ".join(h2_ac) + ";")
    lines.append("  real " + ", ".join(h3_ac) + ";")
    lines.append("  real y0_ac, y1_ac, y2_ac, Cgs, Cgd, Cgg;")
    lines.append("  real gi;")
    lines.append("  analog begin")
    lines.append("    x0 = l; x1 = nfin; x2 = eot; x3 = V(g,s); x4 = V(d,s); x5 = V(s,gnd); x6 = V(b,gnd);")

    # ---- DC path ----
    for i in range(7):
        lines.append(f"    xs_dc_{i} = (x{i} - {_va_real(mean_dc[i])}) / {_va_real(scale_dc[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1_dc[i, j])}*xs_dc_{j}" for j in range(7)]) + f" + {_va_real(b1_dc[i])}"
        lines.append(f"    {h1_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_dc[i])}; {h1_dc[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2_dc[i, j])}*{h1_dc[j]}" for j in range(16)]) + f" + {_va_real(b2_dc[i])}"
        lines.append(f"    {h2_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_dc[i])}; {h2_dc[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3_dc[i, j])}*{h2_dc[j]}" for j in range(32)]) + f" + {_va_real(b3_dc[i])}"
        lines.append(f"    {h3_dc[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_dc[i])}; {h3_dc[i]} = gi;")
    lines.append(f"    y0_dc = " + " + ".join([f"{_va_real(w4_dc[0, j])}*{h3_dc[j]}" for j in range(8)]) + f" + {_va_real(b4_dc[0])};")
    _sid_dc = _va_real(dc_data['scale_id'])
    _mid_dc = _va_real(dc_data['mean_id'])
    lines.append(f"    y = y0_dc * {_sid_dc} + {_mid_dc};")
    if device_type == 'n':
        lines.append(f"    // id_log = y0_dc * {_sid_dc} + {_mid_dc};")
        lines.append("    // id_raw = exp(id_log) ;")
        lines.append("    //if (id_raw < 1e-30) id_raw = 1e-30;")
        lines.append("    // IV inject with optional tanh factor (原 Id 路径已弃用)")
        lines.append("    // if (enable_tanh_iv >= 0.5) begin")
        lines.append("    //   vds_eps = 0.02;")
        lines.append("    //   vds_eff = tanh(V(d,s)/vds_eps);")
        lines.append("    //   Id_si = id_raw * vds_eff;")
        lines.append("    // end else begin")
        lines.append("    //   Id_si = id_raw;")
        lines.append("    // end")
        lines.append("    // PMOS: vsd_eff = tanh(V(s,d)/vds_eps); Id_si = -id_raw * vsd_eff;")
    lines.append("    Id_si = V(d,s) * exp(y);")
    lines.append("    I(d,s) <+ Id_si;")
    if device_type == 'n':
        lines.append("    // [default off] NMOS junction diode: uncomment with junction params above")
        lines.append("    // if (enable_junction > 0.5) begin")
        lines.append("    //   Vbd = V(b,d); Vbs = V(b,s);")
        lines.append("    //   if (Vbd > 0) I(b,d) <+ Isj * (limexp(Vbd / (Nj * 0.02585)) - 1.0);")
        lines.append("    //   if (Vbs > 0) I(b,s) <+ Isj * (limexp(Vbs / (Nj * 0.02585)) - 1.0);")
        lines.append("    // end")
    else:
        lines.append("    // [default off] gate_exp: Id_si = -exp(y)*gate_exp; uncomment with VTH_EFF above")
        lines.append("    // gate_exp = exp((-V(g,s) - VTH_EFF) / (OFF_GAIN * VTSCALE));")
        lines.append("    // if (gate_exp > 1.0) gate_exp = 1.0;")
        lines.append("    // [default off] PMOS junction diode: uncomment with junction params above")
        lines.append("    // if (enable_junction > 0.5) begin")
        lines.append("    //   Vdb = V(d,b); Vsb = V(s,b);")
        lines.append("    //   if (Vdb > 0) I(d,b) <+ Isj * (limexp(Vdb / (Nj * 0.02585)) - 1.0);")
        lines.append("    //   if (Vsb > 0) I(s,b) <+ Isj * (limexp(Vsb / (Nj * 0.02585)) - 1.0);")
        lines.append("    // end")

    # ---- AC path ----
    for i in range(7):
        lines.append(f"    xs_ac_{i} = (x{i} - {_va_real(mean_ac[i])}) / {_va_real(scale_ac[i])};")
    for i in range(16):
        s = " + ".join([f"{_va_real(w1_ac[i, j])}*xs_ac_{j}" for j in range(7)]) + f" + {_va_real(b1_ac[i])}"
        lines.append(f"    {h1_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h1_ac[i])}; {h1_ac[i]} = gi;")
    for i in range(32):
        s = " + ".join([f"{_va_real(w2_ac[i, j])}*{h1_ac[j]}" for j in range(16)]) + f" + {_va_real(b2_ac[i])}"
        lines.append(f"    {h2_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h2_ac[i])}; {h2_ac[i]} = gi;")
    for i in range(8):
        s = " + ".join([f"{_va_real(w3_ac[i, j])}*{h2_ac[j]}" for j in range(32)]) + f" + {_va_real(b3_ac[i])}"
        lines.append(f"    {h3_ac[i]} = {s};")
        lines.append(f"    gi = {_gelu_va(h3_ac[i])}; {h3_ac[i]} = gi;")
    mean_ac_y = np.array(ac_data['mean_y']).ravel()
    scale_ac_y = np.array(ac_data['scale_y']).ravel()
    mean_ac_y = np.resize(mean_ac_y, 3)
    scale_ac_y = np.resize(scale_ac_y, 3)
    lines.append(f"    y0_ac = " + " + ".join([f"{_va_real(w4_ac[0, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[0])};")
    lines.append(f"    y1_ac = " + " + ".join([f"{_va_real(w4_ac[1, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[1])};")
    lines.append(f"    y2_ac = " + " + ".join([f"{_va_real(w4_ac[2, j])}*{h3_ac[j]}" for j in range(8)]) + f" + {_va_real(b4_ac[2])};")
    lines.append(f"    Cgs = y0_ac * {_va_real(scale_ac_y[0])} + {_va_real(mean_ac_y[0])};")
    lines.append(f"    Cgd = y1_ac * {_va_real(scale_ac_y[1])} + {_va_real(mean_ac_y[1])};")
    lines.append(f"    Cgg = y2_ac * {_va_real(scale_ac_y[2])} + {_va_real(mean_ac_y[2])};")
    lines.append("    if (Cgs < 1e-30) Cgs = 1e-30;")
    lines.append("    if (Cgd < 1e-30) Cgd = 1e-30;")
    if device_type == 'n':
        lines.append("    // numerical damping for TRAP (purely numerical, not physical)")
        lines.append("    I(g,s) <+ Cgs * ddt(V(g,s));")
        lines.append("    I(g,d) <+ Cgd * ddt(V(g,d));")
        lines.append("    I(g,s) <+ 1e-6 * V(g,s);")
        lines.append("    I(g,d) <+ 1e-6 * V(g,d);")
    else:
        lines.append("    // I = dQ/dt (charge-conserving); Q_gs = Cgs*V(g,s), Q_gd = Cgd*V(g,d)")
        lines.append("    I(g,s) <+ ddt(Cgs * V(g,s));")
        lines.append("    I(g,d) <+ ddt(Cgd * V(g,d));")
    lines.append("    // Output Cgs,Cgd,Cgg in F (SI) for HSPICE .print; plot script converts F->fF")
    lines.append("    V(cgs_out) <+ Cgs;")
    lines.append("    V(cgd_out) <+ Cgd;")
    lines.append("    V(cgg_out) <+ Cgg;")
    lines.append("  end")
    lines.append("endmodule")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def _emit_params_inc(data, inc_path):
    """将导出数据写成 .inc 参数表（仅参数，供主 .va  include）。"""
    lines = []
    kind = data['kind']
    mean_X = np.array(data['mean_X'])
    scale_X = np.array(data['scale_X'])
    w1, b1 = np.array(data['w1']), np.array(data['b1'])
    w2, b2 = np.array(data['w2']), np.array(data['b2'])
    w3, b3 = np.array(data['w3']), np.array(data['b3'])
    w4, b4 = np.array(data['w4']), np.array(data['b4'])

    lines.append('// Auto-generated parameter include')
    for i in range(7):
        lines.append(f'`define MEAN_X_{i} {_va_real(mean_X[i])}')
        lines.append(f'`define SCALE_X_{i} {_va_real(scale_X[i])}')
    for i in range(16):
        for j in range(7):
            lines.append(f'`define W1_{i}_{j} {_va_real(w1[i, j])}')
        lines.append(f'`define B1_{i} {_va_real(b1[i])}')
    for i in range(32):
        for j in range(16):
            lines.append(f'`define W2_{i}_{j} {_va_real(w2[i, j])}')
        lines.append(f'`define B2_{i} {_va_real(b2[i])}')
    for i in range(8):
        for j in range(32):
            lines.append(f'`define W3_{i}_{j} {_va_real(w3[i, j])}')
        lines.append(f'`define B3_{i} {_va_real(b3[i])}')
    if kind == 'dc':
        for j in range(8):
            lines.append(f'`define W4_0_{j} {_va_real(w4[0, j])}')
            lines.append(f'`define W4_1_{j} {_va_real(w4[1, j])}')
            lines.append(f'`define W4_2_{j} {_va_real(w4[2, j])}')
        lines.append(f'`define B4_0 {_va_real(b4[0])}')
        lines.append(f'`define B4_1 {_va_real(b4[1])}')
        lines.append(f'`define B4_2 {_va_real(b4[2])}')
        lines.append(f'`define MEAN_ID {_va_real(data["mean_id"])}')
        lines.append(f'`define SCALE_ID {_va_real(data["scale_id"])}')
        lines.append(f'`define MEAN_GM {_va_real(data["mean_gm"])}')
        lines.append(f'`define SCALE_GM {_va_real(data["scale_gm"])}')
        lines.append(f'`define MEAN_GDS {_va_real(data["mean_gds"])}')
        lines.append(f'`define SCALE_GDS {_va_real(data["scale_gds"])}')
        lines.append(f'`define IV_LOG_THRESH {_va_real(data["iv_log_threshold"])}')
        lines.append(f'`define SIGN_ID {_va_real(data["sign"])}')
    else:
        mean_y = np.array(data['mean_y']).ravel()
        scale_y = np.array(data['scale_y']).ravel()
        mean_y = np.resize(mean_y, 3)
        scale_y = np.resize(scale_y, 3)
        for j in range(8):
            lines.append(f'`define W4_0_{j} {_va_real(w4[0, j])}')
            lines.append(f'`define W4_1_{j} {_va_real(w4[1, j])}')
            lines.append(f'`define W4_2_{j} {_va_real(w4[2, j])}')
        lines.append(f'`define B4_0 {_va_real(b4[0])}')
        lines.append(f'`define B4_1 {_va_real(b4[1])}')
        lines.append(f'`define B4_2 {_va_real(b4[2])}')
        lines.append(f'`define MEAN_CGS {_va_real(mean_y[0])}')
        lines.append(f'`define MEAN_CGD {_va_real(mean_y[1])}')
        lines.append(f'`define MEAN_CGG {_va_real(mean_y[2])}')
        lines.append(f'`define SCALE_CGS {_va_real(scale_y[0])}')
        lines.append(f'`define SCALE_CGD {_va_real(scale_y[1])}')
        lines.append(f'`define SCALE_CGG {_va_real(scale_y[2])}')

    with open(inc_path, 'w') as f:
        f.write('\n'.join(lines))


def json_to_inc(out_dir):
    """从已存在的 JSON 重新生成 .inc、分体 .va 与整合 N/P 型 .va（无需 PyTorch）。"""
    for base in ['params_dc_n', 'params_dc_p', 'params_ac_n', 'params_ac_p']:
        json_path = os.path.join(out_dir, base + '.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        _emit_params_inc(data, os.path.join(out_dir, base + '.inc'))
        if data['kind'] == 'dc':
            dtype = data['device_type']
            _generate_va_iv_full(data, os.path.join(out_dir, f"nnmodel_iv_dc_{dtype}.va"))
        else:
            dtype = data['device_type']
            _generate_va_cv_full(data, os.path.join(out_dir, f"nnmodel_cv_ac_{dtype}.va"))
        print(f"已从 JSON 生成: {base}.inc 及对应 .va")
        for dtype in ['n', 'p']:
            dc_json = os.path.join(out_dir, f'params_dc_{dtype}.json')
            ac_json = os.path.join(out_dir, f'params_ac_{dtype}.json')
            if os.path.isfile(dc_json) and os.path.isfile(ac_json):
                with open(dc_json) as f:
                    dc_data = json.load(f)
                with open(ac_json) as f:
                    ac_data = json.load(f)
            va_path = os.path.join(out_dir, f'nnmodel_{dtype}_type.va')
            _generate_va_integrated(dc_data, ac_data, va_path, dtype)
            print(f"已生成整合 VA: nnmodel_{dtype}_type.va")
    print("从 JSON 生成 .inc 与 .va 完成。")
