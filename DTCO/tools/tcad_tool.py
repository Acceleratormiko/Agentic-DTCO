import asyncio
import gc
import json
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from openai import AsyncOpenAI

from .common import (
    calc_width_from_json,
    extract_iv_metrics,
    format_error_response,
    format_success_response,
    validate_design_variables,
    is_safe_temp_dir,
)
from internbootcamp.src.base_tool import BaseTool
from internbootcamp.src.openai_tool_schema import OpenAIFunctionToolSchema
# from internbootcamp.common.compat import rollout_trace_op


class TCADTool(BaseTool):
    """TCAD 仿真工具：生成器件结构、运行 TCAD 仿真、提取 IdVg 指标"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.semaphore = asyncio.Semaphore(config.get("semaphore_limit", 2))
        self.device_type = config.get("device_type", "n-type")
        # LLM 配置
        self.model_path = config.get("model", "s1-qwen14b-30k-baseline")
        self.base_url = config.get("base_url", "http://100.101.23.77:2025/v1")
        self.api_key = config.get("api_key", "api-key")
        self.temperature = config.get("temperature", 0.0)
        self.base_work_dir_parent = config.get("base_work_dir_parent", "/tmp")
        # code_cases 目录路径
        current_file = Path(__file__).resolve()
        self.code_cases_dir = current_file.parent.parent / "configs" / "tool_configs" / "code_cases"
    
    def _code_generator_prompt(self, params: dict[str, Any], device_type: str = None) -> tuple[str, str, str]:
        """生成 SDE、SDEVICE IdVg 和 CV 的 prompt"""
        if device_type is None:
            device_type = self.device_type
        device_type = device_type.lower()
        
        # 根据器件类型选择 device_desc
        if device_type == "p-type":
            gaafet_device_desc = "p-type nanosheet FET array"
            finfet_device_desc = "p-type FinFET array"
            device_desc = gaafet_device_desc
        else:
            gaafet_device_desc = "n-type nanosheet FET array"
            finfet_device_desc = "n-type FinFET array"
            device_desc = finfet_device_desc
        
        gaafet_sde_prompt = (
            f"Can you provide an sde script for a 3D {device_desc} with the following explicit parameters:\n"
            f"- Lg = {params.get('Lg', 0.016)} um (gate length)\n"
            f"- Lext = {params.get('Lext', 0.006)} um (extension length)\n"
            f"- Lsd = {params.get('Lsd', 0.01)} um (source/drain length)\n"
            f"- Fh = {params.get('Fh', 0.0035)} um (TODO: what is this?)\n"
            f"- Fwt = {params.get('Fwt', 0.0035)} um (TODO: what is this?)\n"
            f"- NSwt = {params.get('NSwt', 0.028)} um (nanosheet width)\n"
            f"- NSt = {params.get('NSt', 0.006)} um (nanosheet thickness)\n"
            f"- NSbuf = {params.get('NSbuf', 0.004)} um (buffer spacing between sheets)\n"
            f"- theta = {params.get('theta', 3)} degree (tilt angle of the structure)\n"
            f"- Tox = {params.get('Tox', 0.0011)} um (gate oxide thickness)\n"
            f"- num_of_ns = {params.get('num_of_ns', 3)} (number of nanosheets)\n"
            f"- SD_Doping = {params.get('SD_Doping', 6e20)} cm-3 (source/drain doping concentration)\n"
            f"- SDE_Doping = {params.get('SDE_Doping', 5e19)} cm-3 (source/drain extension doping)\n"
            f"- Sub_Doping = {params.get('Sub_Doping', 1e17)} cm-3 (substrate doping)\n"
        )
        
        finfet_sde_prompt = (
            f"Generate a default sde structure file for a 3D {device_desc} with standard design and meshing parameters:\n"
            f"- Lg = {params.get('Lg', 0.018)} um (gate length)\n"
            f"- Lext = {params.get('Lext', 0.005)} um (extension length)\n"
            f"- Lsd = {params.get('Lsd', 0.014)} um (source/drain length)\n"
            f"- Fh = {params.get('Fh', 0.035)} um (TODO: what is this?)\n"
            f"- Fwt = {params.get('Fwt', 0.006)} um (TODO: what is this?)\n"
            f"- Fsp = {params.get('Fsp', 0.03)} um (TODO: what is this?)\n"
            f"- theta = {params.get('theta', 0.82)} degree (tilt angle of the structure)\n"
            f"- ToxLK = {params.get('ToxLK', 5e-4)} um (gate oxide thickness of LK layer)\n"
            f"- ToxHK = {params.get('ToxHK', 1.13e-3)} um (gate oxide thickness of HK layer)\n"
            f"- num_of_fin = {2} (number of fins)\n"
            f"- SD_Doping = {params.get('SD_Doping_Fin', 1e20)} cm-3 (source/drain doping concentration)\n"
            f"- SDE_Doping = {params.get('SDE_Doping_Fin', 1e18)} cm-3 (source/drain extension doping)\n"
            f"- C_Doping = {params.get('C_Doping_Fin', 5e17)} cm-3 (channel doping)\n"
        )
        
        sde_prompt = finfet_sde_prompt
        
        sdevice_prompt = (
            f"I'd like a TCAD script for a standard Sentaurus Device (sdevice) IdVg simulation of a 3D {device_desc}, using following parameters:\n"
            "- Vgmin = -0.1 V (minimum gate voltage)\n"
            "- Vgmax = 0.8 V (maximum gate voltage)\n"
            "- VDD   = 0.7 V (supply voltage)\n"
        )
        
        sdevice_cv_prompt = (
            f"I'd like to simulate a 3D {device_desc} using a Sentaurus Device (sdevice) CV simulation with an explicit gate voltage of Vgs = 0.8 V.\n"
            f"Could you generate the corresponding TCAD script?"
        )
        
        return sde_prompt, sdevice_prompt, sdevice_cv_prompt
    
    async def _tcad_code_generation(self, user_prompt: str) -> str:
        """调用 LLM 生成 TCAD 脚本"""
        system_prompt = (
            "You are an expert in TCAD simulation for semiconductor devices, especially using Synopsys Sentaurus tools. Your task is to generate accurate and high-quality simulation scripts based on given descriptions."
        )
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        response = await client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def _load_code_from_template(
        self, 
        code_type: str, 
        device_type: str, 
        design_variables: dict[str, Any]
    ) -> str:
        """从 code_cases 目录加载模板文件并替换参数
        
        Args:
            code_type: 代码类型，可选值: "sde", "iv", "cv"
            device_type: 器件类型，"n-type" 或 "p-type"
            design_variables: 设计参数字典，用于替换模板中的参数
            
        Returns:
            替换参数后的代码字符串
        """
        device_type = device_type.lower()
        if device_type not in ["n-type", "p-type"]:
            raise ValueError(f"Invalid device_type: {device_type}. Must be 'n-type' or 'p-type'")
        
        # 确定文件映射
        file_mapping = {
            "n-type": {
                "sde": "sde.cmd",
                "iv": "sdevice_iv.cmd",
                "cv": "sdevice_cv.cmd"
            },
            "p-type": {
                "sde": "sde.cmd",
                "iv": "sdevice_iv.cmd",
                "cv": "sdevice_cv.cmd"
            }
        }
        
        if code_type not in file_mapping[device_type]:
            raise ValueError(f"Invalid code_type: {code_type}. Must be one of {list(file_mapping[device_type].keys())}")
        
        # 构建文件路径
        template_file = self.code_cases_dir / device_type.replace("-", "") / file_mapping[device_type][code_type]
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        # 读取模板文件
        template_content = template_file.read_text(encoding="utf-8")
        
        # 替换参数
        # 对于 SDE 文件，需要替换的参数包括：Lg, Lext, Lsd, Fh, Fwt, Fsp, theta, ToxLK, ToxHK, num_of_fin, SD_doping, SD_extension_doping, C_doping, Sub_Doping
        # 对于 SDEVICE 文件，通常不需要替换参数，但可能需要替换文件名等
        
        if code_type == "sde":
            # 替换 SDE 参数
            replacements = {
                "Lg": design_variables.get("Lg", 0.018),
                "Lext": design_variables.get("Lext", 0.005),
                "Lsd": design_variables.get("Lsd", 0.014),
                "Fh": design_variables.get("Fh", 0.035),
                "Fwt": design_variables.get("Fwt", 0.006),
                "Fsp": design_variables.get("Fsp", 0.03),
                "theta": design_variables.get("theta", 0.82),
                "ToxLK": design_variables.get("ToxLK", 0.5e-3),
                "ToxHK": design_variables.get("ToxHK", 1.13e-3),
                "num_of_fin": design_variables.get("num_of_fin", 2),
            }
            
            # 处理掺杂浓度
            if device_type == "n-type":
                replacements["SD_doping"] = design_variables.get("SD_Doping", 1e20)
                replacements["SD_extension_doping"] = design_variables.get("SDE_Doping", 1e19)
                replacements["C_doping"] = design_variables.get("C_Doping", 5e17)
                replacements["Sub_Doping"] = design_variables.get("C_Doping", 5e17)
            else:  # p-type, but temporarily the same as n-type
                replacements["SD_doping"] = design_variables.get("SD_Doping", 1e20)
                replacements["SD_extension_doping"] = design_variables.get("SDE_Doping", 1e19)
                replacements["C_doping"] = design_variables.get("C_Doping", 5e17)
                replacements["Sub_Doping"] = design_variables.get("C_Doping", 5e17)
            
            # 执行替换
            for key, value in replacements.items():
                # 匹配 (define key value) 格式
                pattern = rf'\(define\s+{key}\s+[^)]+\)'
                replacement = f'(define {key} {value})'
                template_content = re.sub(pattern, replacement, template_content)
        
        return template_content
    
    async def _run_sde(self, sde_code: str, sde_prefix: str, work_dir: Path) -> int:
        """运行 SDE 生成器件结构"""
        output_path = work_dir / f"{sde_prefix}.cmd"
        processed_code = "\n".join(
            (lambda lines: (lines[:-1] + [f'(sde:build-mesh "{sde_prefix}")'])
            if lines and lines[-1].startswith('(sde:build-mesh') else lines)
            (sde_code.strip().splitlines())
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_code)
        
        if shutil.which("sde") is None:
            raise FileNotFoundError("找不到可执行文件'sde'")
        
        cmd = ["sde", "-e", "-l", str(output_path)]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line_bytes in process.stdout:
            line = line_bytes.decode('utf-8', errors='ignore').strip()
        ret = await process.wait()
        del process
        if ret == 0:
            print(f"[INFO] SDE运行完成✅，临时文件为: {output_path}")
        else:
            raise RuntimeError(f"SDE运行失败❌，错误码: {ret}")
        return ret
    
    def _write_sdevice_par(
        self, 
        work_dir: Path, 
        par_prefix: str, 
        device_type: str = "n-type",
        work_function: float = None
    ) -> None:
        """从模板文件写入 SDEVICE 参数文件并替换 WorkFunction
        
        Args:
            work_dir: 工作目录
            par_prefix: 参数文件前缀
            device_type: 器件类型，"n-type" 或 "p-type"
            work_function: 功函数值，如果为 None 则使用模板中的默认值
        """
        device_type = device_type.lower()
        if device_type not in ["n-type", "p-type"]:
            raise ValueError(f"Invalid device_type: {device_type}. Must be 'n-type' or 'p-type'")
        
        # 构建模板文件路径
        template_file = self.code_cases_dir / device_type.replace("-", "") / "sdevice.par"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        # 读取模板文件
        template_content = template_file.read_text(encoding="utf-8")
        
        # 如果提供了 work_function，替换 WorkFunction 值
        if work_function is not None:
            # 匹配 WorkFunction = value 格式
            pattern = r'WorkFunction\s*=\s*[\d.]+'
            replacement = f'WorkFunction = {work_function}'
            template_content = re.sub(pattern, replacement, template_content)
        
        # 写入参数文件
        par_path = work_dir / f"{par_prefix}.par"
        par_path.write_text(template_content, encoding="utf-8")

    def _save_sdevice_iv_code(
        self,
        sdevice_code: str,
        sdevice_prefix: str,
        sde_prefix: str,
        par_prefix: str,
        work_dir: Path,
    ) -> Path:
        """保存处理后的 SDEVICE IV 脚本"""
        lines = sdevice_code.splitlines()
        processed_lines = []
        last_prefix = None
        prefix_pattern = re.compile(r'NewCurrentPrefix\s*=\s*"([^"]+)"')
        currentplot_pattern = re.compile(r'(CurrentPlot\s*\(.*\))\s*\}')
        inside_file_block = False
        
        for line in lines:
            if prefix_pattern.search(line):
                last_prefix = "result_Sat_"
                processed_lines.append('NewCurrentPrefix = "result_Sat_"')
                continue
            if line.strip().startswith("File{"):
                inside_file_block = True
                processed_lines += [
                    "File{",
                    f'  Grid = "{sde_prefix}_msh.tdr"',
                    f'  Plot = "IdVg_des.tdr"',
                    f'  Parameter = "{par_prefix}.par"',
                    f'  Current = "IdVg_des.plt"',
                    f'  Output = "IdVg_des.log"',
                    "}",
                ]
                continue
            if inside_file_block:
                if line.strip() == "}":
                    inside_file_block = False
                continue
            if (cm := currentplot_pattern.search(line)) and last_prefix:
                indent = re.match(r"(\s*)", line).group(1)
                currentplot_part = cm.group(1)
                processed_lines.append(f"{indent}{currentplot_part}")
                processed_lines.append(
                    f'{indent}Plot( -Loadable fileprefix="{last_prefix}" Nooverwrite Time=(Range=(0 1) Intervals=10))}}'
                )
            else:
                processed_lines.append(line)
        
        output_path = work_dir / f"{sdevice_prefix}.cmd"
        output_path.write_text("\n".join(processed_lines), encoding="utf-8")
        return output_path

    def _save_sdevice_cv_code(
        self,
        sdevice_code: str,
        sdevice_prefix: str,
        sde_prefix: str,
        par_prefix: str,
        work_dir: Path,
    ) -> Path:
        """保存处理后的 SDEVICE CV 脚本"""
        lines = sdevice_code.splitlines()
        processed_lines = []
        prefix_pattern = re.compile(r'NewCurrentPrefix\s*=\s*".*?"')
        file_block_index = 0
        inside_file_block = False
        
        for line in lines:
            if prefix_pattern.search(line):
                indent = re.match(r"(\s*)", line).group(1)
                processed_lines.append(f'{indent}NewCurrentPrefix="result"')
                continue
            if line.strip().startswith("File"):
                file_block_index += 1
                inside_file_block = True
                processed_lines.append("File{")
                if file_block_index == 1:
                    processed_lines.append(f'  Grid      = "{sde_prefix}_msh.tdr"')
                    processed_lines.append('  Plot      = "DC_des.tdr"')
                    processed_lines.append(f'  Parameter = "{par_prefix}.par"')
                    processed_lines.append('  Current   = "DC_des.plt"')
                    processed_lines.append("}")
                elif file_block_index == 2:
                    processed_lines.append('  Output    = "CV_des.log"')
                    processed_lines.append('  ACExtract = "_ac_des.plt"')
                    processed_lines.append("}")
                continue
            if inside_file_block:
                if line.strip() == "}":
                    inside_file_block = False
                continue
            processed_lines.append(line)
        
        output_path = work_dir / f"{sdevice_prefix}.cmd"
        output_path.write_text("\n".join(processed_lines), encoding="utf-8")
        return output_path

    def _read_iv_plt(self, work_dir: str) -> Dict[str, List[float]]:
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

    def _read_cv_plt(self, work_dir: Path, filename: str = "result_ac_des.plt") -> Dict[str, List[float]]:
        """读取 CV .plt 文件"""
        full_path = work_dir / filename
        if not full_path.exists():
            raise FileNotFoundError(f"未在目录中找到 C-V .plt 文件: {full_path}")
        
        headers = []
        all_data_rows = []
        with open(full_path, "r", encoding="utf-8") as f:
            in_datasets = False
            in_data = False
            current_data_block = []
            pending_single_value = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("datasets") and line.endswith("= ["):
                    in_datasets = True
                    continue
                
                if line.startswith("Data {"):
                    in_data = True
                    in_datasets = False
                    continue
                
                if line == "]":
                    in_datasets = False
                    continue
                
                if in_datasets:
                    if line == "]":
                        in_datasets = False
                        continue
                    if (
                        line.startswith("functions")
                        or line.startswith("Frequency")
                        or line.startswith("Voltage")
                        or line.startswith("Admittance")
                        or line.startswith("Capacitance")
                    ):
                        continue
                    processed_headers = [
                        h.strip('"') for h in line.split() if h.startswith('"') and h.endswith('"')
                    ]
                    headers.extend(processed_headers)
                
                if in_data:
                    if line == "}":
                        if current_data_block:
                            if pending_single_value is not None:
                                current_data_block.append(pending_single_value)
                                pending_single_value = None
                            flat_data = " ".join(current_data_block).split()
                            if len(flat_data) == len(headers):
                                all_data_rows.append(flat_data)
                        break
                    
                    parts = line.split()
                    is_freq_line = False
                    if len(parts) == 1:
                        try:
                            freq_val = float(parts[0])
                            if freq_val >= 1e3:
                                is_freq_line = True
                            else:
                                pending_single_value = line
                                continue
                        except ValueError:
                            is_freq_line = False
                    
                    if is_freq_line:
                        if current_data_block:
                            if pending_single_value is not None:
                                current_data_block.append(pending_single_value)
                                pending_single_value = None
                            flat_data = " ".join(current_data_block).split()
                            if len(flat_data) == len(headers):
                                all_data_rows.append(flat_data)
                        current_data_block = [line]
                        pending_single_value = None
                    else:
                        current_data_block.append(line)
        
        vg_key = "v(g)"
        cgg_key = "c(g,g)"
        if vg_key not in headers or cgg_key not in headers:
            raise ValueError(f"Missing required CV headers: {vg_key} or {cgg_key}")
        
        data_dict = {
            "vg_dc": [],
            "gate_gate_Capacitance": [],
        }
        for row_list in all_data_rows:
            if len(row_list) == len(headers):
                row_dict = dict(zip(headers, row_list))
                data_dict["vg_dc"].append(float(row_dict[vg_key]))
                data_dict["gate_gate_Capacitance"].append(float(row_dict[cgg_key]))
        
        return data_dict
    
    async def _run_sdevice(
        self,
        sdevice_code: str,
        sdevice_prefix: str,
        sde_prefix: str,
        par_prefix: str,
        work_dir: Path,
        type: str,
        device_type: str = "n-type",
        work_function: float = None
    ) -> int:
        """运行 SDEVICE 仿真"""
        if type == "iv":
            output_path = self._save_sdevice_iv_code(
                sdevice_code, sdevice_prefix, sde_prefix, par_prefix, work_dir
            )
        else:
            output_path = self._save_sdevice_cv_code(
                sdevice_code, sdevice_prefix, sde_prefix, par_prefix, work_dir
            )
        self._write_sdevice_par(work_dir, par_prefix, device_type, work_function)
        
        if shutil.which("sdevice") is None:
            raise FileNotFoundError("找不到可执行文件'sdevice'")
        
        msh = work_dir / f"{sde_prefix}_msh.tdr"
        if not msh.exists():
            raise FileNotFoundError(f"缺少网格文件: {msh}")
        
        cmd = ["sdevice", "--exit-on-failure", str(output_path)]
        print(f"[INFO] 正在运行 sdevice (工作目录: {work_dir})")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line_bytes in process.stdout:
            line = line_bytes.decode('utf-8', errors='ignore').strip()
        ret = await process.wait()
        del process
        if ret == 0:
            print("[INFO] sdevice运行完成✅")
        else:
            raise RuntimeError(f"sdevice运行失败❌，错误码: {ret}")
        return ret
    
    async def _run_single_device_simulation(
        self,
        design_variables: dict[str, Any],
        device_type: str,
        work_dir_path: Path,
        work_function: float = None,
    ) -> dict[str, Any]:
        """运行单个器件类型的 TCAD 仿真"""
        # 从模板文件加载代码
        try:
            tcad_sde_code = self._load_code_from_template("sde", device_type, design_variables)
        except Exception as e:
            raise RuntimeError(f"SDE code loading failed: {str(e)}")
        
        try:
            tcad_iv_code = self._load_code_from_template("iv", device_type, design_variables)
        except Exception as e:
            raise RuntimeError(f"SDEVICE IV code loading failed: {str(e)}")

        try:
            tcad_cv_code = self._load_code_from_template("cv", device_type, design_variables)
        except Exception as e:
            raise RuntimeError(f"SDEVICE CV code loading failed: {str(e)}")
        
        # 运行仿真
        sde_prefix = "mos_structure"
        sdevice_iv_prefix = "mos_iv_simulation"
        sdevice_cv_prefix = "mos_cv_simulation"
        par_prefix = "sdevice"
        sdevice_task_list = []
        
        # 运行 SDE
        sde_ret = await self._run_sde(tcad_sde_code, sde_prefix, work_dir_path)
        if sde_ret != 0:
            raise RuntimeError("SDE simulation failed.")
        
        # 创建 SDEVICE 任务
        iv_sdevice_task = self._run_sdevice(
            tcad_iv_code, sdevice_iv_prefix, sde_prefix, par_prefix, work_dir_path, 
            type="iv", device_type=device_type, work_function=work_function
        )
        sdevice_task_list.append(iv_sdevice_task)
        cv_sdevice_task = self._run_sdevice(
            tcad_cv_code, sdevice_cv_prefix, sde_prefix, par_prefix, work_dir_path, 
            type="cv", device_type=device_type, work_function=work_function
        )
        sdevice_task_list.append(cv_sdevice_task)
        
        # 运行 SDEVICE
        iv_ret, cv_ret = await asyncio.gather(*sdevice_task_list, return_exceptions=True)
        
        # 处理异常返回值
        if isinstance(iv_ret, Exception):
            raise RuntimeError(f"SDEVICE IV simulation failed: {iv_ret}")
        if isinstance(cv_ret, Exception):
            raise RuntimeError(f"SDEVICE CV simulation failed: {cv_ret}")
        
        result = {
            "iv_result": 'success' if iv_ret == 0 else 'failed',
            "cv_result": 'success' if cv_ret == 0 else 'failed',
            "work_dir": str(work_dir_path),
        }
        
        # 解析 CV 结果
        if cv_ret == 0:
            try:
                cv_data = self._read_cv_plt(work_dir_path, "result_ac_des.plt")
                if cv_data and len(cv_data.get("vg_dc", [])) > 0:
                    cv_data_json = work_dir_path / "cv_data.json"
                    with open(cv_data_json, "w", encoding="utf-8") as f:
                        json.dump(cv_data, f, indent=2)
                    del cv_data
            except Exception as e:
                print(f"[WARNING] CV 数据解析失败: {e}")
        
        # 解析 IdVg 结果
        if iv_ret == 0:
            width = calc_width_from_json(design_variables)
            simu_result = self._read_iv_plt(str(work_dir_path))
            mos_type_str = "n" if device_type == "n-type" else "p"
            ext_result = extract_iv_metrics(
                str(work_dir_path),
                simu_result,
                0.75,    # VDD
                mos_type_str,
                2,      # order
                "IdVg",
                width,
            )
            result["metrics"] = {
                "Ion": ext_result["Ion"],
                "Ioff": ext_result["Ioff"],
                "SS": ext_result["SS"],
                "VDD": ext_result["VDD"],
                "width_um": width,
            }
            del simu_result
            del ext_result
        else:
            raise RuntimeError("SDEVICE IV simulation failed.")
        
        gc.collect()
        return result
    
    # @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """执行 TCAD 仿真并返回结果
        
        支持从 parameters 中读取 device_type，可选值：
        - "n-type": 仅处理 N-type 器件
        - "p-type": 仅处理 P-type 器件
        - "both": 同时处理 N-type 和 P-type 器件
        如果未提供，则使用 config 中的 device_type（默认为 "n-type"）
        
        参数结构（parameters）：
        - device_type (str, optional): 器件类型，默认为 config 中的值
        - ntype_variables (dict): N-type 的设计参数
        - ptype_variables (dict): P-type 的设计参数
        - ntype_work_function (float, optional): N-type 器件的功函数值，用于替换 sdevice.par 中的 WorkFunction
        - ptype_work_function (float, optional): P-type 器件的功函数值，用于替换 sdevice.par 中的 WorkFunction
        
        参数使用策略：
        - 单一类型模式（"n-type"）：必须提供 ntype_variables
        - 单一类型模式（"p-type"）：必须提供 ptype_variables
        - "both" 模式：必须提供 ntype_variables 和 ptype_variables，两个器件类型使用完全独立的参数
        
        示例：
        # 单一类型 N-type
        {
            "device_type": "n-type",
            "ntype_variables": {"Lg": 0.016, "SD_Doping": 6e20, ...},
            "ntype_work_function": 4.5 
        }
        
        # 单一类型 P-type
        {
            "device_type": "p-type",
            "ptype_variables": {"Lg": 0.016, "SD_Doping_P": 6e20, ...},
            "ptype_work_function": 4.7
        }
        
        # both 模式，N-type 和 P-type 使用完全独立的参数
        {
            "device_type": "both",
            "ntype_variables": {
                "Lg": 0.016,
                "Lext": 0.006,
                "SD_Doping": 6e20,
                "SDE_Doping": 5e19,
                ...
            },
            "ptype_variables": {
                "Lg": 0.018,
                "Lext": 0.006,
                "SD_Doping_P": 6e20,
                "SDE_Doping_P": 5e19,
                ...
            },
            "ntype_work_function": 4.5, 
            "ptype_work_function": 4.7
        }
        """
        async with self.semaphore:
            try:
                # Step 1: 获取 device_type（优先从 parameters 读取，否则使用 config）
                device_type = parameters.get("device_type", {})
                if not device_type:
                    return format_error_response("Missing required parameter: device_type"), 0, {}
                device_type = device_type.lower()
                
                # 验证 device_type 值
                if device_type not in ["n-type", "p-type", "both"]:
                    return format_error_response(
                        f"Invalid device_type: {device_type}. Must be 'n-type', 'p-type', or 'both'."
                    ), 0, {}
                
                # Step 2: 解析 variables
                # 所有模式都使用 ntype_variables 或 ptype_variables# 验证参数
                process_type = self._instance_dict[instance_id].get("process_type", "finfet_7nm")
                if device_type == "both":
                    # both 模式：需要独立的 ntype_variables 和 ptype_variables
                    ntype_variables = parameters.get("ntype_variables", {})
                    ntype_variables.update({"ntype_work_function":parameters.get("ntype_work_function", None)})
                    ptype_variables = parameters.get("ptype_variables", {})
                    ptype_variables.update({"ptype_work_function":parameters.get("ptype_work_function", None)})
                    
                    if not ntype_variables:
                        return format_error_response(
                            "For 'both' mode, 'ntype_variables' is required."
                        ), 0, {}
                    
                    if not ptype_variables:
                        return format_error_response(
                            "For 'both' mode, 'ptype_variables' is required."
                        ), 0, {}
                    
                    is_valid_ntype, error_msg_ntype = validate_design_variables(ntype_variables, process_type=process_type, device_type='ntype')
                    if not is_valid_ntype:
                        return format_error_response(f"N-type variables validation failed: {error_msg_ntype}"), 0, {}
                    
                    is_valid_ptype, error_msg_ptype = validate_design_variables(ptype_variables, process_type=process_type, device_type='ptype')
                    if not is_valid_ptype:
                        return format_error_response(f"P-type variables validation failed: {error_msg_ptype}"), 0, {}
                    
                    design_variables_ntype = ntype_variables
                    design_variables_ptype = ptype_variables
                elif device_type == "n-type":
                    # N-type 单一类型：使用 ntype_variables
                    ntype_variables = parameters.get("ntype_variables", {})
                    ntype_variables.update({"ntype_work_function":parameters.get("ntype_work_function", None)})
                    if not ntype_variables:
                        return format_error_response(
                            "For 'n-type' mode, 'ntype_variables' is required."
                        ), 0, {}
                    
                    is_valid, error_msg = validate_design_variables(ntype_variables, process_type=process_type, device_type='ntype')
                    if not is_valid:
                        return format_error_response(f"N-type variables validation failed: {error_msg}"), 0, {}
                    
                    design_variables_ntype = ntype_variables
                    design_variables_ptype = {}  # 不使用
                elif device_type == "p-type":
                    # P-type 单一类型：使用 ptype_variables
                    ptype_variables = parameters.get("ptype_variables", {})
                    ptype_variables.update({"ptype_work_function":parameters.get("ptype_work_function", None)})
                    if not ptype_variables:
                        return format_error_response(
                            "For 'p-type' mode, 'ptype_variables' is required."
                        ), 0, {}
                    
                    is_valid, error_msg = validate_design_variables(ptype_variables, process_type=process_type, device_type='ptype')
                    if not is_valid:
                        return format_error_response(f"P-type variables validation failed: {error_msg}"), 0, {}
                    
                    design_variables_ntype = {}  # 不使用
                    design_variables_ptype = ptype_variables
                
                # Step 3: 获取 work_function 参数（区分 n-type 和 p-type）
                ntype_work_function = parameters.get("ntype_work_function", None)
                ptype_work_function = parameters.get("ptype_work_function", None)
                
                if ntype_work_function is not None:
                    try:
                        ntype_work_function = float(ntype_work_function)
                    except (ValueError, TypeError):
                        return format_error_response(
                            f"Invalid ntype_work_function: {ntype_work_function}. Must be a float."
                        ), 0, {}
                
                if ptype_work_function is not None:
                    try:
                        ptype_work_function = float(ptype_work_function)
                    except (ValueError, TypeError):
                        return format_error_response(
                            f"Invalid ptype_work_function: {ptype_work_function}. Must be a float."
                        ), 0, {}
                
                # Step 4: 在指定 base 目录下创建临时目录（每次调用独立目录，避免并发冲突）
                Path(self.base_work_dir_parent).mkdir(parents=True, exist_ok=True)
                base_work_dir = tempfile.mkdtemp(prefix="tcad_", dir=self.base_work_dir_parent)
                base_work_dir_path = Path(base_work_dir)
                
                # 更新 instance_dict
                if "temp_dir" not in self._instance_dict[instance_id]:
                    self._instance_dict[instance_id]["temp_dir"] = []
                self._instance_dict[instance_id]["temp_dir"].append(str(base_work_dir))
                
                # Step 5: 根据 device_type 执行仿真
                if device_type == "both":
                    # 同时处理 N-type 和 P-type（并发执行）
                    ntype_work_dir = base_work_dir_path / "ntype"
                    ptype_work_dir = base_work_dir_path / "ptype"
                    ntype_work_dir.mkdir(parents=True, exist_ok=True)
                    ptype_work_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 定义包装函数以处理异常
                    async def run_ntype_simulation():
                        try:
                            result = await self._run_single_device_simulation(
                                design_variables_ntype, "n-type", ntype_work_dir, ntype_work_function
                            )
                            return result
                        except Exception as e:
                            print(f"[ERROR] N-type 仿真失败: {e}")
                            traceback.print_exc()
                            return {
                                "status": "failed",
                                "error": str(e),
                                'metrics': {}
                            }
                    
                    async def run_ptype_simulation():
                        try:
                            result = await self._run_single_device_simulation(
                                design_variables_ptype, "p-type", ptype_work_dir, ptype_work_function
                            )
                            return result
                        except Exception as e:
                            print(f"[WARNING] P-type 仿真失败: {e}")
                            traceback.print_exc()
                            return {
                                "status": "failed",
                                "error": str(e),
                                'metrics': {}
                            }
                    
                    # 并发执行 N-type 和 P-type 仿真
                    ntype_result, ptype_result = await asyncio.gather(
                        run_ntype_simulation(),
                        run_ptype_simulation(),
                        return_exceptions=True
                    )
                    
                    # 处理 N-type 结果
                    if isinstance(ntype_result, Exception):
                        print(f"[WARNING] N-type 仿真异常: {ntype_result}")
                    
                    # 处理 P-type 结果
                    if isinstance(ptype_result, Exception):
                        print(f"[WARNING] P-type 仿真异常: {ptype_result}")
                    
                    # 保存 device_metrics 到各自的子目录
                    # N-type metrics
                    if isinstance(ntype_result, dict) and ntype_result.get("metrics"):
                        ntype_metrics_file = ntype_work_dir / "device_metrics.json"
                        ntype_metrics = {
                            "Ion": ntype_result["metrics"].get("Ion"),
                            "Ioff": ntype_result["metrics"].get("Ioff"),
                            "SS": ntype_result["metrics"].get("SS"),
                            "VDD": ntype_result["metrics"].get("VDD"),
                            "width_um": ntype_result["metrics"].get("width_um"),
                            "device_type": "n-type",
                            "variables": design_variables_ntype,
                        }
                        with open(ntype_metrics_file, "w", encoding="utf-8") as f:
                            json.dump(ntype_metrics, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] N-type device metrics 已保存到: {ntype_metrics_file}")
                    
                    # P-type metrics
                    if isinstance(ptype_result, dict) and ptype_result.get("metrics"):
                        ptype_metrics_file = ptype_work_dir / "device_metrics.json"
                        ptype_metrics = {
                            "Ion": ptype_result["metrics"].get("Ion"),
                            "Ioff": ptype_result["metrics"].get("Ioff"),
                            "SS": ptype_result["metrics"].get("SS"),
                            "VDD": ptype_result["metrics"].get("VDD"),
                            "width_um": ptype_result["metrics"].get("width_um"),
                            "device_type": "p-type",
                            "variables": design_variables_ptype,
                        }
                        with open(ptype_metrics_file, "w", encoding="utf-8") as f:
                            json.dump(ptype_metrics, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] P-type device metrics 已保存到: {ptype_metrics_file}")
                    
                    # 构建响应数据
                    response_data = {
                        "work_dir": str(base_work_dir),
                        "device_type": "both",
                        "ntype_result": ntype_result,
                        "ptype_result": ptype_result,
                    }
                    
                    # 保存 CV 数据路径（如果存在）
                    if ntype_result.get("cv_data_path"):
                        response_data["ntype_cv_data_path"] = ntype_result["cv_data_path"]
                    if ptype_result and ptype_result.get("cv_data_path"):
                        response_data["ptype_cv_data_path"] = ptype_result["cv_data_path"]
                    
                else:
                    # 处理单一器件类型
                    # 为了与 both 模式目录结构对齐：单一模式也落在子目录 ntype/ 或 ptype/
                    design_variables_single = design_variables_ntype if device_type == "n-type" else design_variables_ptype
                    work_function_single = ntype_work_function if device_type == "n-type" else ptype_work_function

                    subdir_name = "ntype" if device_type == "n-type" else "ptype"
                    single_work_dir = base_work_dir_path / subdir_name
                    single_work_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        result = await self._run_single_device_simulation(
                            design_variables_single, device_type, single_work_dir, work_function_single
                        )
                    except Exception as e:
                        print(f"[ERROR] {device_type} 仿真失败: {e}")
                        return format_error_response(
                            f"{device_type} simulation failed: {str(e)}"
                        ), 0, {}

                    # 保存 device_metrics 到文件，格式如 device_metrics.json
                    if result.get("metrics"):
                        device_metrics_file = single_work_dir / "device_metrics.json"
                        device_metrics = {
                            "Ion": result["metrics"].get("Ion"),
                            "Ioff": result["metrics"].get("Ioff"),
                            "SS": result["metrics"].get("SS"),
                            "VDD": result["metrics"].get("VDD"),
                            "width_um": result["metrics"].get("width_um"),
                            "device_type": device_type,
                            "variables": design_variables_single,
                        }
                        with open(device_metrics_file, "w", encoding="utf-8") as f:
                            json.dump(device_metrics, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] Device metrics 已保存到: {device_metrics_file}")

                    response_data = {
                        "work_dir": str(base_work_dir),
                        "device_type": device_type,
                        "result": result,
                    }
            
                response = format_success_response(response_data)
                reward = 1.0
                metrics = response_data
                gc.collect()
                return response, reward, metrics
                
            except Exception as e:
                return format_error_response(f"Unexpected error: {str(e)}"), 0, {}
    
    async def release(self, instance_id: str, **kwargs) -> bool:
            """释放资源并清理 temp_dir"""
            try:
                if instance_id in self._instance_dict:
                    temp_dir_str_list = self._instance_dict[instance_id].get("temp_dir")
                    if not temp_dir_str_list:
                        return True
                    for temp_dir_str in temp_dir_str_list:
                        if temp_dir_str:
                            # 安全检查后清理
                            if is_safe_temp_dir(temp_dir_str):
                                try:
                                    shutil.rmtree(temp_dir_str)
                                    print(f"[INFO] 已清理临时目录: {temp_dir_str}")
                                except Exception as e:
                                    print(f"[WARNING] 清理临时目录失败: {temp_dir_str}, 错误: {e}")
                            else:
                                print(f"[WARNING] 跳过不安全的目录清理: {temp_dir_str}")
                    del self._instance_dict[instance_id]
                gc.collect()
                return True
            except Exception as e:
                print(f"[ERROR] Release 失败: {e}")
                return False
