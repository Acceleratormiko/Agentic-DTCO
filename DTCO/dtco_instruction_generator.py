import math
import random
from typing import List, Dict

from internbootcamp.src.base_instruction_generator import BaseInstructionGenerator
from internbootcamp.utils.encrypted_yaml_loader import load_yaml_config_from_path


def _param_range_to_tuple(param_range):
    try:
        a, b = param_range
    except:
        raise ValueError("Parameter range must be a list or tuple of two elements.")
    try:
        a = float(a); b = float(b)
    except:
        raise ValueError("Parameter range elements must be convertible to float.")
    return (a, b)

class DTCOInstructionGenerator(BaseInstructionGenerator):
    def __init__(self,
                 process_type: str = "finfet_7nm",
                 seed: int = 42,
                 prompt_template_path: str = None,
                 enable_fix_params: bool = False,
                 enable_mix_difficulty: bool = False,
                 fix_params_list: List[str] = [],
                 initial_variables: Dict = None,
                 targets: Dict = None,
                 **kwargs):
        super().__init__()
        self.data_source = f"bootcamp/{self.__class__.__name__.replace('InstructionGenerator', '')}"
        self.process_type = process_type
        self.seed = int(seed)
        random.seed(self.seed)
        self.enable_mix_difficulty = bool(enable_mix_difficulty)
        
        # 读取参数范围：优先使用 initial_variables，否则从 kwargs 中读取（向后兼容）
        if initial_variables is not None:
            self.param_ranges = {}
            for param_name, param_range in initial_variables.items():
                self.param_ranges[param_name] = _param_range_to_tuple(param_range)
        else:
            # 向后兼容：从 kwargs 中直接读取参数
            self.param_ranges = {}
            for param_name, param_value in kwargs.items():
                if param_name not in ['Ion_target', 'Ioff_target', 'SS_target']:
                    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
                        self.param_ranges[param_name] = _param_range_to_tuple(param_value)
                    elif isinstance(param_value, (int, float)):
                        # 单个值，转换为范围（用于向后兼容）
                        self.param_ranges[param_name] = (float(param_value), float(param_value))
        
        # 读取目标值：优先使用 targets，否则从 kwargs 中读取（向后兼容）
        if targets is not None:
            # 器件目标 - 支持 List 类型
            self.Ion_target = targets.get('Ion_target')
            self.Ioff_target = targets.get('Ioff_target')
            self.SS_target = targets.get('SS_target')
            # 电路目标 - 支持 List 类型
            self.delay_scale_s = targets.get('delay_scale_s')
            self.power_scale_W = targets.get('power_scale_W')
        else:
            raise ValueError("必须提供 targets ")
        
        if self.Ion_target is None or self.Ioff_target is None or self.SS_target is None:
            raise ValueError("必须提供 targets 或 kwargs 中的 Ion_target, Ioff_target, SS_target")
        
        self._load_prompt_template(prompt_template_path)
        self.enable_fix_params = enable_fix_params
        self.fix_params_list = fix_params_list

    def _load_prompt_template(self, prompt_template_path: str):
        try:
            self.prompt_template = load_yaml_config_from_path(prompt_template_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt template from {prompt_template_path}: {e}")
        
    def case_generator(self) -> Dict:
        identity = {"process_type": self.process_type}
        
        # 动态生成所有参数的值
        for param_name, param_range in self.param_ranges.items():
            min_val, max_val = param_range
            
            # 如果范围的两个值相同，直接使用固定值
            if min_val == max_val:
                param_value = min_val
            # 对于掺杂浓度参数（参数名包含 "Doping"），使用对数均匀分布
            elif "Doping" in param_name:
                param_value = 10 ** random.uniform(math.log10(min_val), math.log10(max_val))
            # 其他参数使用均匀分布
            else:
                param_value = random.uniform(min_val, max_val)
            
            # 根据参数类型选择格式化方式
            if "Doping" in param_name:
                identity[param_name] = float(f"{param_value:.3e}")
                identity[f"{param_name}_min"] = float(f"{min_val:.3e}")
                identity[f"{param_name}_max"] = float(f"{max_val:.3e}")
            elif param_name.startswith("Tox"):
                # 对于氧化层厚度参数，使用科学计数法以保持精度
                identity[param_name] = float(f"{param_value:.3e}")
                identity[f"{param_name}_min"] = float(f"{min_val:.3e}")
                identity[f"{param_name}_max"] = float(f"{max_val:.3e}")
            elif param_name == "num_of_fin" or isinstance(param_value, int) or (isinstance(param_value, float) and param_value.is_integer()):
                identity[param_name] = int(param_value)
                identity[f"{param_name}_min"] = int(min_val)
                identity[f"{param_name}_max"] = int(max_val)
            else:
                identity[param_name] = float(f"{param_value:.3f}")
                identity[f"{param_name}_min"] = float(f"{min_val:.3f}")
                identity[f"{param_name}_max"] = float(f"{max_val:.3f}")
        
        # 处理器件目标值 - 支持 List 类型（enable_mix_difficulty 时随机选择）
        ion_target_value = self._resolve_target_value(self.Ion_target)
        ioff_target_value = self._resolve_target_value(self.Ioff_target)
        ss_target_value = self._resolve_target_value(self.SS_target)
        
        identity["Ion_target"] = float(f"{ion_target_value:.2e}")
        identity["Ioff_target"] = float(f"{ioff_target_value:.2e}")
        identity["SS_target"] = float(f"{ss_target_value:.2e}")
        
        identity["delay_scale_s"] = float(f"{self.delay_scale_s:.2e}")
        identity["power_scale_W"] = float(f"{self.power_scale_W:.2e}")
        identity["delay_target"] = float(f"{self.delay_scale_s * 0.7:.2e}")
        identity["power_target"] = float(f"{self.power_scale_W * 0.7:.2e}")
        
        return identity
    
    def _resolve_target_value(self, target_value):
        """解析目标值，支持 List 类型。当 enable_mix_difficulty 为 True 且目标值为列表时，随机选择一个值。"""
        if self.enable_mix_difficulty and isinstance(target_value, list):
            if not target_value:
                raise ValueError("目标值列表不能为空")
            return float(random.choice(target_value))
        elif isinstance(target_value, list):
            # 如果不启用 mix_difficulty 但目标值是列表，使用第一个值
            if not target_value:
                raise ValueError("目标值列表不能为空")
            return float(target_value[0])
        return float(target_value)

    def prompt_func(self, identity: Dict) -> str:
        # 只保留与当前 case 相关的动态信息（指标要求和参数范围）
        # 流程相关的、不变的 prompt 已迁移到 agent config 的 system_prompt 中
        
        device_requirements = (
            "【器件性能指标要求】\n"
            "1. 开态电流 Ion > {Ion_target} A/um\n"
            "2. 关态电流 Ioff < {Ioff_target} A/um\n"
            "3. 亚阈值摆幅 SS < {SS_target} mV/dec\n"
        ).format_map(identity)

        circuit_requirements = (
            "【电路性能指标要求】\n"
            "- 重点关注 HSPICE verified 的 延迟/功耗指标\n"
            "- 延迟指标：平均延迟 < {delay_target} s； 功耗指标：平均功耗 < {power_target} W\n"
            "- 在满足器件指标后，优先优化电路性能；在满足电路指标前，请勿停止优化\n"
        ).format_map(identity)

        parameter_list = (
            "可调节的参数列表、初始值和范围如下，括号中左边是最小值，右边是最大值：\n"
            "\"Lg\": {Lg}, ({Lg_min}, {Lg_max}) # 栅极长度 (um)\n"
            "\"Lext\": {Lext}, ({Lext_min}, {Lext_max}) # 源漏延伸长度 (um)\n"
            "\"Lsd\": {Lsd}, ({Lsd_min}, {Lsd_max}) # 源漏长度 (um)\n"
            "\"Fh\": {Fh}, ({Fh_min}, {Fh_max}) # Fin高度 (um)\n"
            "\"Fwt\": {Fwt}, ({Fwt_min}, {Fwt_max}) # Fin顶部宽度 (um)\n"
            "\"theta\": {theta}, ({theta_min}, {theta_max}) # 结构倾斜角度 (度)\n"
            "\"ToxLK\": {ToxLK}, ({ToxLK_min}, {ToxLK_max}) # LK层栅氧化层厚度 (um)\n"
            "\"ToxHK\": {ToxHK}, ({ToxHK_min}, {ToxHK_max}) # HK层栅氧化层厚度 (um)\n"
            "\"SD_Doping\": {SD_Doping}, ({SD_Doping_min}, {SD_Doping_max}) # 源漏掺杂浓度 (cm-3)\n"
            "\"SDE_Doping\": {SDE_Doping}, ({SDE_Doping_min}, {SDE_Doping_max}) # 源漏延伸掺杂浓度 (cm-3)\n"
            "\"C_Doping\": {C_Doping}, ({C_Doping_min}, {C_Doping_max}) # 沟道掺杂浓度 (cm-3)\n"
            "\"ntype_work_function\": 建议初值 4.40 eV，范围 [4.15, 4.65]\n"
            "\"ptype_work_function\": 建议初值 4.90 eV，范围 [4.65, 5.15]\n"
        ).format_map(identity)

        # 处理固定参数逻辑
        if self.enable_fix_params and self.fix_params_list:
            lines = parameter_list.split('\n')
            filtered_lines = []
            fixed_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped: continue
                if stripped.startswith('"'):
                    is_fixed = any(f'"{param}"' in line for param in self.fix_params_list)
                    if is_fixed: fixed_lines.append(stripped)
                    else: filtered_lines.append(stripped)
                else:
                    filtered_lines.append(stripped)
            
            parameter_list = '\n'.join(filtered_lines)
            if fixed_lines:
                parameter_list += "\n固定的参数列表如下（不可调整）：\n" + '\n'.join(fixed_lines)

        prompt = f"{device_requirements}\n{circuit_requirements}\n{parameter_list}"
        return prompt
