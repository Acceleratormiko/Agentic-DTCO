"""
共享工具函数和常量，供 DTCO 三个工具使用
"""
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def is_number(s):
    """检查字符串是否为数字"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False



# 设计空间定义
GAAFET_2NM_DESIGN_SPACE = {
    "Lg": {"min": 0.01, "max": 0.028},
    "Lext": {"min": 0.004, "max": 0.015},
    "Lsd": {"min": 0.006, "max": 0.014},
    "NSwt": {"min": 0.026, "max": 0.032},
    "NSt": {"min": 0.002, "max": 0.01},
    "NSbuf": {"min": 0.002, "max": 0.008},
    "Tox": {"min": 0.001, "max": 0.003},
    "SD_Doping": {"min": 6e19, "max": 6e21},
    "SDE_Doping": {"min": 5e18, "max": 5e20},
    "Sub_Doping": {"min": 1e16, "max": 1e18}
}

FINFET_7NM_DESIGN_SPACE = {
    "Lg": {"min": 0.014, "max": 0.028},
    "Lext": {"min": 0.004, "max": 0.1},
    "Lsd": {"min": 0.01, "max": 0.022},
    "Fh": {"min": 0.03, "max": 0.06},
    "Fwt": {"min": 0.005, "max": 0.009},
    "theta": {"min": 0.5, "max": 4.0},
    "ToxLK": {"min": 4.00e-04, "max": 1.50e-03},
    "ToxHK": {"min": 1.00e-03, "max": 1.50e-03},
    "SD_Doping": {"min": 6e19, "max": 6e21},
    "SDE_Doping": {"min": 6e17, "max": 6e20},
    "C_Doping": {"min": 1e16, "max": 1e18},
}

FINFET_7NM_DESIGN_SPACE_NTYPE = FINFET_7NM_DESIGN_SPACE.copy()
FINFET_7NM_DESIGN_SPACE_NTYPE.update({"ntype_work_function": {"min": 4.15, "max": 4.65}})
FINFET_7NM_DESIGN_SPACE_PTYPE = FINFET_7NM_DESIGN_SPACE.copy()
FINFET_7NM_DESIGN_SPACE_PTYPE.update({"ptype_work_function": {"min": 4.65, "max": 5.15}})

DESIGN_SPACE = FINFET_7NM_DESIGN_SPACE
DESIGN_SPACE_MAP = {
    'finfet_7nm': {'ntype': FINFET_7NM_DESIGN_SPACE_NTYPE, 'ptype': FINFET_7NM_DESIGN_SPACE_PTYPE},
    'gaa_2nm': GAAFET_2NM_DESIGN_SPACE,
}

def validate_design_variables(variables: dict, process_type: str, device_type: str) -> tuple[bool, str]:
    """
    验证设计变量是否在有效范围内
    
    Returns:
        (is_valid, error_message)
    """
    if process_type not in DESIGN_SPACE_MAP:
        return False, f"Unknown process type: {process_type}. Please provide supported process types only."
    else:
        design_space = DESIGN_SPACE_MAP[process_type][device_type]
    if not len(variables) == len(design_space):
        raise ValueError("设计变量数量与设计空间不匹配")
    for key, value in variables.items():
        if key not in design_space:
            return False, f"Unknown design parameter '{key}': {value}. Please provide supported parameters only."
        if not is_number(str(value)):
            return False, f"Invalid value for {key}: {value}."
        if not (design_space[key]["min"] <= float(value) <= design_space[key]["max"]):
            return False, f"Value for {key} out of range: {value}. Please provide values within the range of {design_space[key]['min']} and {design_space[key]['max']}."
    return True, ""



def calc_width_from_json(params: dict[str, Any]) -> float:
    fh = float(params.get("Fh"))
    fwt = float(params.get("Fwt"))
    num_of_fin = int(1)
    w = (2 * fh + fwt) * num_of_fin
    return w


def read_iv_plt(work_dir: str) -> Dict[str, List[float]]:
    """
    从工作目录读取 .plt 文件并解析数据
    Args:
        work_dir: 工作目录路径
    Returns:
        data_dict: 包含变量名和数据的字典
    """
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"工作目录不存在: {work_dir}")
    candidates = []
    for fn in os.listdir(work_dir):
        if fn.lower().endswith(".plt"):
            lower = fn.lower()
            if ("result" in lower) and ("sat" in lower):
                full_path = os.path.join(work_dir, fn)
                if os.path.isfile(full_path):
                    candidates.append(full_path)
    if not candidates:
        raise FileNotFoundError("未在目录中找到文件名包含'result'且'sat'的.plt文件")
    filename = max(candidates, key=lambda p: os.path.getmtime(p))
    name_tmp = []
    data_tmp = []
    variable_name = []
    data_result = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            if line.split():
                if line.strip().split()[0] == "datasets":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            if value.strip('"') != "]":
                                name_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "]":
                            break
                if line.strip().split()[0] == "Data":
                    while line:
                        line = f.readline()
                        for value in line.split():
                            data_tmp.append(value.strip('"'))
                        if line.strip().split()[-1] == "}":
                            break
            line = f.readline()
    for i in range(len(name_tmp)):
        if i == 0:
            variable_name.append(name_tmp[i])
        if (i != 0) & (i % 2 == 0):
            variable_name.append(name_tmp[i-1] + "_" + name_tmp[i])
    time_tmp = []
    for i in range(len(data_tmp[:-1])):
        time_tmp.append(float(data_tmp[i]))
        if len(time_tmp) == len(variable_name):
            data_result.append(time_tmp)
            time_tmp = []
    data_result.insert(0, variable_name)
    header = data_result[0]
    rows = data_result[1:]
    data_dict = {col: [row[i] for row in rows] for i, col in enumerate(header)}
    return data_dict


def extract_iv_metrics(
    work_dir,
    data_dict,
    VDD: float,
    mos_type: str,
    order: int,
    output_prefix: str,
    w_um: float
) -> dict:
    """
    提取 Id–Vg 指标并返回数据（电流归一化为 A/um）
    
    Args:
        work_dir: 保留参数（向后兼容），不再落盘
        data_dict: 至少包含 'gate_OuterVoltage' 和 'drain_TotalCurrent'
        VDD: 电源电压 (float)
        mos_type: 'n' or 'p'
        order: SS 的 decade 数 (int)
        output_prefix: 保留参数（向后兼容），不再落盘
        w_um: 归一化宽度（单位 um）。所有电流输出将以 A/um 表示
        
    Returns:
        dict = {
            "Ioff", "Ion", "SS",
            "VDD", "mos_type", "order_decade", "width_um",
            "curve_Vg_Id": {"units": ["V","A/um"], "data": [[Vg, Id], ...]}
        }
    """
    Vg_vec = np.array(data_dict["gate_OuterVoltage"], dtype=float)
    Id_vec_A = np.array(data_dict["drain_TotalCurrent"], dtype=float)
    mos_type = (mos_type or "").lower()
    if mos_type.startswith("n"):
        # nMOS: Vg 从小到大排序（0 -> 正电压）
        sort_indices = np.argsort(Vg_vec)
    else:
        # pMOS: Vg 从大到小排序（0 -> 负电压）
        sort_indices = np.argsort(Vg_vec)[::-1]

    Vg_vec = Vg_vec[sort_indices]
    Id_vec_A = Id_vec_A[sort_indices]
    Id_vec = np.abs(Id_vec_A) / float(w_um)
    Id_vec = np.clip(Id_vec, 1e-30, None)
    
    # ====== 筛选单调区域 ======
    # 对于 nMOS 和 pMOS，都筛选电流单调递增的区域（从小到大）
    # 这样第一个 mask 都是 Vg=0 的值
    mono_mask = np.concatenate([[True], Id_vec[1:] > Id_vec[:-1]])
    Vg_mono = Vg_vec[mono_mask]
    Id_mono = Id_vec[mono_mask]  # A/um
    # === 保证 Vg 单调递增，供 np.interp 使用 ===
    if mos_type.lower().startswith("p"):
        sort_idx = np.argsort(Vg_mono)
        Vg_mono = Vg_mono[sort_idx]
        Id_mono = Id_mono[sort_idx]
    # nMOS 不动，默认已经是 0 → VDD
    
    log10_Id_mono = np.log10(Id_mono)
    
    # 供 Vg→log10(Id) 插值（Ioff/Ion）
    Vg_for_Id_interp = Vg_mono.copy()
    log10_Id_for_Id_interp = log10_Id_mono.copy()

    # 供 log10(Id)→Vg 反插值（SS）
    # 由于排序后电流都是从小到大，log10_Id 也是从小到大，不需要反转
    log10_Id_sorted = log10_Id_mono
    Vg_for_logId_interp = Vg_mono
    
    # ====== Ioff / Ion（在 log10(Id) 域按 Vg 插值）======
    # Ioff: n和p都是抽取0V的电流
    # Ion: nMOS在VDD处抽取，pMOS在-VDD处抽取
    log10_Ioff = np.interp(0.0, Vg_for_Id_interp, log10_Id_for_Id_interp)
    if mos_type.startswith("n"):
        # nMOS: Ion在VDD处抽取
        vg_ion = VDD
    else:
        # pMOS: Ion在-VDD处抽取
        vg_ion = -VDD
    
    log10_Ion = np.interp(vg_ion, Vg_for_Id_interp, log10_Id_for_Id_interp)
    Ioff_A_per_um = float(10.0 ** log10_Ioff)
    Ion_A_per_um  = float(10.0 ** log10_Ion)

    # ====== SS 计算（不随宽度变化）======
    # SS = (Vg_target - Vg_off) / order * 1000
    # Vg_off = 0V（n和p都是从Vg=0开始）
    target_log10 = log10_Ioff + float(order)
    if target_log10 >= log10_Ion:
        SS_mV_dec = 1000.0
        Vg_target = float(Vg_for_logId_interp[-1])  # 供标注参考
    else:
        # === 反插值：log10(Id) -> Vg ===
        log10_Id_sorted = log10_Id_sorted.copy()
        Vg_for_logId_interp = Vg_for_logId_interp.copy()

        if mos_type.lower().startswith("p"):
            # pMOS: 保证 log10(Id) 单调递增
            sort_idx = np.argsort(log10_Id_sorted)
            log10_Id_sorted = log10_Id_sorted[sort_idx]
            Vg_for_logId_interp = Vg_for_logId_interp[sort_idx]
        Vg_target = float(np.interp(target_log10, log10_Id_sorted, Vg_for_logId_interp))
        # 使用abs确保SS为正数（对于pMOS，Vg_target可能是负数）
        SS_mV_dec = abs((Vg_target - 0.0) / float(order)) * 1000.0
    SS_mV_dec = float(SS_mV_dec)

    return {
        "Ioff": Ioff_A_per_um,
        "Ion": Ion_A_per_um,
        "SS": SS_mV_dec,
        "VDD": float(VDD),
        "mos_type": mos_type,
        "order_decade": int(order),
        "width_um": float(w_um)
    }


def is_safe_temp_dir(temp_dir: str, allowed_prefix: str = "tcad_") -> bool:
    """
    检查临时目录路径是否安全（用于清理时验证）
    
    Args:
        temp_dir: 临时目录路径
        allowed_prefix: 允许的目录名前缀
        
    Returns:
        是否安全可删除
    """
    path = Path(temp_dir)
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    # 检查目录名是否以允许的前缀开头
    if not path.name.startswith(allowed_prefix):
        return False
    # 检查是否在临时目录下（通过检查是否包含 tmp 或 temp 路径段）
    parts = path.parts
    has_tmp = any(p in ["tmp", "temp", "var"] for p in parts)
    if not has_tmp:
        return False
    return True


def format_error_response(error_msg: str) -> str:
    """格式化错误响应为 JSON 字符串"""
    return json.dumps({"error": error_msg}, ensure_ascii=False)


def format_success_response(data: dict) -> str:
    """格式化成功响应为 JSON 字符串"""
    return json.dumps(data, ensure_ascii=False)

