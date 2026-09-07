"""
update: 2026-02-04
    说明：
        1）HspiceTool需要提供semaphore参数，用于控制并发访问Hspice的数量，建议设置为电路类型数量
        2）HspiceTool需要提供base_dir参数，实际仿真：从base_dir找到对应电路的子目录 => 复制到tmp_dir => 执行hspice => 从tmp_dir中.mt0文件中提取结果 => 构造context返回给agent
        3）HspiceTool需要提供circuit_types参数，类型为List[str]，用于表示需要进行hspice仿真的circuit
        4）HspiceTool需要提供circuit_ratios参数，类型为List[float]，用于表示hspice仿真得到.mt0文件提取出的delay和power metrics的归一化比例，其实就是构建电路锁包含的inv与and总数，注意circuit_ratios和circuit_types要一一对应
"""
import os
import shutil
import asyncio
from pathlib import Path
from typing import Any

from internbootcamp.src.base_tool import BaseTool
from internbootcamp.src.openai_tool_schema import OpenAIFunctionToolSchema
# from internbootcamp.common.compat import rollout_trace_op

class HspiceTool(BaseTool):
    TIMEOUT = 1200
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        assert config is not None, "HspiceTool __init__需要提供config参数"
        assert tool_schema is not None, "HspiceTool __init__需要提供tool_schema参数"
        try:
            semaphore = config.get("semaphore")
        except Exception as e:
            raise ValueError("HspiceTool __init__失败，config参数缺少semaphore键") from e
        """
        HspiceTool需要在config中提供base_dir参数
        base_dir的目录结构如下：
        base_dir/
            ├── AND/
            |   ├── and_subckt_nn.inc
            |   └── run_hspice.sp
            ├── BUF/
            |   ├── buf_subckt_nn_cv.inc
            |   └── run_hspice.sp
            ├── INV/
            |   ├── inv_subckt_nn_cv.inc
            |   └── run_hspice.sp
            ├── NAND/
            |   ├── nand_subckt_nn.inc
            |   └── run_hspice.sp
            ├── NOR/
            |   ├── nor_subckt_nn.inc
            |   └── run_hspice.sp
            ├── OR/
            |   ├── or_subckt_nn.inc
            |   └── run_hspice.sp
            ├── XNOR/
            |   ├── xnor_subckt_nn.inc
            |   └── run_hspice.sp
            └── XOR/
                ├── xor_subckt_nn.inc
                └── run_hspice.sp
        """
        try:
            base_dir = config.get("base_dir")
        except Exception as e:
            raise ValueError("HspiceTool __init__失败，config参数缺少base_dir键") from e
        if not Path(base_dir) or not Path(base_dir).is_dir():
            raise ValueError(f"HspiceTool __init__中base_dir目录不存在：{base_dir}")
        try:
            circuit_types = config.get("circuit_types")
        except Exception as e:
            raise ValueError("HspiceTool __init__失败，config参数缺少circuit_types键") from e
        try:
            circuit_ratios = config.get("circuit_ratios")
        except Exception as e:
            raise ValueError("HspiceTool __init__失败，config参数缺少circuit_ratios键") from e
        assert isinstance(circuit_types, list), "HspiceTool __init__中circuit_types必须为List[str]类型"
        assert isinstance(circuit_ratios, list), "HspiceTool __init__中circuit_ratios必须为List[float]类型"
        assert semaphore == len(circuit_types), "HspiceTool __init__中semaphore值需要等于circuit_types数量，否则会有资源浪费或阻塞"
        assert len(circuit_types) == len(circuit_ratios), "HspiceTool __init__中circuit_types和circuit_ratios数量需要一致"
        for circuit_type in circuit_types:
            circuit_dir = Path(base_dir) / circuit_type
            if not circuit_dir or not circuit_dir.is_dir():
                raise ValueError(f"HspiceTool __init__中base_dir目录下缺少电路子目录：{circuit_dir}")
        super().__init__(config, tool_schema)
        self.semaphore = asyncio.Semaphore(semaphore)
        self.base_dir = base_dir
        self.circuit_types = circuit_types
        self.circuit_ratios = circuit_ratios
        self.inc_path = str(Path(base_dir) / "va_model_wrapper_new.inc")
        if not Path(self.inc_path).is_file():
            raise ValueError(f"HspiceTool __init__中缺少wrapper文件：{self.inc_path}")

    def _get_ratio(self, idx: int, circuit_type: str, circuit_ratios=None) -> float:
        """支持 list 或 dict 两种 ratio 配置形式。"""
        ratios = self.circuit_ratios if circuit_ratios is None else circuit_ratios
        if isinstance(ratios, dict):
            ratio = ratios.get(circuit_type, 1.0)
        else:
            ratio = ratios[idx] if idx < len(ratios) else 1.0
        ratio = float(ratio)
        return ratio if ratio != 0.0 else 1.0

    # @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        try:
            vdd = float(parameters.get("vdd"))
            cl = float(parameters.get("cl"))
            slew = float(parameters.get("slew"))
            ntype_va_path = parameters.get("ntype_va_path")
            ptype_va_path = parameters.get("ptype_va_path")
            work_dir = parameters.get("work_dir")
        except Exception:
            return "执行HspiceTool失败，输入参数格式错误", 0, {}

        if not all([ntype_va_path, ptype_va_path, work_dir]):
            return "执行HspiceTool失败，缺少必要参数（ntype_va_path/ptype_va_path/work_dir）", 0, {}

        work_dir = Path(work_dir)
        if not work_dir.exists():
            return "执行HspiceTool失败，work_dir目录不存在", 0, {}

        circuit_types = list(self.circuit_types)
        if isinstance(self.circuit_ratios, dict):
            circuit_ratios = dict(self.circuit_ratios)
        else:
            circuit_ratios = list(self.circuit_ratios)

        # 兼容旧配置：若未显式列出 MAC4bit，则在本次执行中补上默认 ratio
        if "MAC4bit" not in circuit_types:
            circuit_types.append("MAC4bit")
            if isinstance(circuit_ratios, dict):
                circuit_ratios["MAC4bit"] = 1.0
            else:
                circuit_ratios.append(1.0)

        # 首先将 base_dir 的电路子目录复制到 work_dir 下，并放置模型文件
        for circuit_type in circuit_types:
            circuit_dir = Path(self.base_dir) / circuit_type
            dest_dir = work_dir / circuit_type
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(circuit_dir, dest_dir)
            shutil.copy(ntype_va_path, dest_dir / "nnmodel_n_type.va")
            shutil.copy(ptype_va_path, dest_dir / "nnmodel_p_type.va")
            shutil.copy(self.inc_path, dest_dir / "va_model_wrapper_new.inc")

        # 替换各子目录 netlist 参数
        for circuit_type in circuit_types:
            sp_file_path = work_dir / circuit_type / "run_hspice.sp"
            self.replace_sp_parameters(sp_file_path, vdd, cl, slew)

        # 并行执行多个 hspice run_hspice.sp
        tasks = [
            self.execute_circuit(work_dir, circuit_type)
            for circuit_type in circuit_types
        ]
        # 允许单个电路失败：使用 return_exceptions=True 收集结果
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 记录每个电路的运行状态（成功 / 失败及失败原因）
        run_status: dict[str, Any] = {}
        fatal_errors: list[str] = []
        for circuit_type, result in zip(circuit_types, raw_results):
            if isinstance(result, Exception):
                # 这里一般是环境级/编程级错误，视为致命错误，直接失败返回
                fatal_errors.append(f"{circuit_type}: {result}")
            elif isinstance(result, dict):
                run_status[result.get("circuit_type", circuit_type)] = result
            else:
                # 理论上不会发生，兜底记录
                run_status[circuit_type] = {
                    "circuit_type": circuit_type,
                    "success": False,
                    "error": f"未知返回类型: {type(result)}",
                }

        if fatal_errors:
            # 如果是 HSPICE_BIN 缺失、协程内部编程错误等致命问题，直接返回失败
            joined = "; ".join(fatal_errors)
            return f"执行HspiceTool失败，HSPICE 环境或工具内部错误：{joined}", 0, {}

        # 提取不同子目录的 .mt0 文件 delay 和 power，同时保留失败电路的错误信息
        response = ""
        metrics: dict[str, Any] = {}
        metrics["circuits"] = {}

        any_success = False
        any_failure = False

        for idx, circuit_type in enumerate(circuit_types):
            status = run_status.get(circuit_type, {})
            entry: dict[str, Any] = {}

            # 如果该电路在 execute_circuit 阶段就已经失败，则不再尝试解析 .mt0
            if not status or not status.get("success", False):
                error_msg = status.get("error") or "HSPICE 运行失败，未能生成有效的 .mt0 文件"
                entry["error"] = error_msg
                metrics["circuits"][circuit_type] = entry
                any_failure = True

                # MAC4bit 仍然保持静默：不拼进 response，只在 metrics 中体现错误
                if circuit_type != "MAC4bit":
                    response += f"电路类型: {circuit_type}, 仿真失败: {error_msg}\n"
                continue

            # 对于 execute_circuit 成功的电路，尝试解析 .mt0
            mt0_path = work_dir / circuit_type / "run_hspice.mt0"
            try:
                delay, power = self.parse_mt0_metrics(mt0_path)
                ratio = self._get_ratio(idx, circuit_type, circuit_ratios)
                norm_delay = delay / ratio
                norm_power = power / ratio

                entry["delay"] = norm_delay
                entry["power"] = norm_power
                any_success = True

                # 对于 MAC4bit，只在 metrics 中记录，不输出到 response
                if circuit_type != "MAC4bit":
                    response += (
                        f"电路类型: {circuit_type}, 延迟: {norm_delay} s, "
                        f"功耗: {norm_power} W\n"
                    )
            except Exception as e:
                # 单个电路解析失败：记录错误，但不影响其它电路
                error_msg = f"解析 .mt0 文件失败：{e}"
                entry["error"] = error_msg
                any_failure = True
                if circuit_type != "MAC4bit":
                    response += f"电路类型: {circuit_type}, 仿真结果解析失败: {error_msg}\n"

            metrics["circuits"][circuit_type] = entry

        # 统一处理文本 response
        if not response:
            if any_success and any_failure:
                response = "部分 HSPICE 仿真成功，部分失败（详细见 metrics.circuits）。"
            elif any_success:
                response = "HSPICE 仿真已完成。"
            else:
                response = "所有 HSPICE 仿真均失败（详细见 metrics.circuits）。"

        # 至少有一个电路成功解析出 delay/power 时给予正奖励
        reward = 1 if any_success else 0
        return response, reward, metrics

    def replace_sp_parameters(self, sp_file_path: Path, vdd: float, cl: float, slew: float):
        with open(sp_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        has_gshunt = False
        first_option_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            lower = stripped.lower()
            if stripped.startswith('.param VDD'):
                lines[i] = f'.param VDD   = {vdd}\n'
            elif stripped.startswith('.param CL'):
                lines[i] = f'.param CL    = {cl}\n'
            elif stripped.startswith('.param slew'):
                lines[i] = f'.param slew  = {slew}\n'
            elif stripped.startswith('.param TR'):
                lines[i] = f'.param TR    = {slew}\n'
            elif stripped.startswith('.param TF'):
                lines[i] = f'.param TF    = {slew}\n'
            if lower.startswith('.option'):
                if first_option_idx is None:
                    first_option_idx = i
                if 'gshunt' in lower:
                    has_gshunt = True
        if not has_gshunt:
            gshunt_option = '.option gshunt=1e-10\n'
            if first_option_idx is not None:
                lines.insert(first_option_idx + 1, gshunt_option)
            else:
                lines.insert(0, gshunt_option)
        with open(sp_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def parse_mt0_metrics(self, mt0_path: Path):
        """从 .mt0 文件中解析 delay / power。

        兼容两种列名格式：
        1）tpd_avg（延迟）、p_dyn_avg（功耗）
        2）tpd_avg（延迟）、pavg_1c（功耗）
        """
        mt0_path = Path(mt0_path)
        if not mt0_path.exists():
            raise FileNotFoundError(f".mt0 文件未找到: {mt0_path}")

        with mt0_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError(f".mt0 文件为空: {mt0_path}")

        # 先找到包含 tpd_avg 的表头行
        header_tokens = None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("$"):
                continue
            tokens = stripped.split()
            if "tpd_avg" in tokens:
                header_tokens = tokens
                break

        # 如果没找到标准表头，则退回到老逻辑报错，避免静默解析错误
        if header_tokens is None:
            raise ValueError(f".mt0 文件格式不正确，未找到 tpd_avg 表头: {mt0_path}")

        try:
            delay_idx = header_tokens.index("tpd_avg")
        except ValueError as e:
            raise ValueError(f".mt0 表头中缺少 tpd_avg 列: {mt0_path}") from e

        # 功耗列优先级：p_dyn_avg > pavg_1c
        power_col_candidates = ["p_dyn_avg", "pavg_1c"]
        power_idx = None
        power_col_name = None
        for col in power_col_candidates:
            if col in header_tokens:
                power_idx = header_tokens.index(col)
                power_col_name = col
                break
        if power_idx is None:
            raise ValueError(
                f".mt0 表头中未找到功耗列（期望 p_dyn_avg 或 pavg_1c）: {mt0_path}"
            )

        # 从文件末尾向前找一行有效的数据行
        data_tokens = None
        min_len = max(delay_idx, power_idx) + 1
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("$") or stripped.upper().startswith(".TITLE"):
                continue
            tokens = stripped.split()
            if len(tokens) >= min_len:
                data_tokens = tokens
                break

        if data_tokens is None:
            raise ValueError(f".mt0 文件中未找到合法数据行: {mt0_path}")

        raw_delay = data_tokens[delay_idx]
        raw_power = data_tokens[power_idx]

        # 处理 HSPICE 仿真失败的情况
        if raw_delay.lower() in ("failed", "fail"):
            raise ValueError(
                f"HSPICE 仿真失败，delay(tpd_avg) 测量值为 '{raw_delay}'，"
                f"请检查输入参数是否合理。mt0 文件: {mt0_path}"
            )
        if raw_power.lower() in ("failed", "fail"):
            raise ValueError(
                f"HSPICE 仿真失败，power({power_col_name}) 测量值为 '{raw_power}'，"
                f"请检查输入参数是否合理。mt0 文件: {mt0_path}"
            )

        try:
            # 兼容 HSPICE 的 'd'/'D' 科学计数法（如 1.5d-11 → 1.5e-11）
            delay = float(raw_delay.replace("d", "e").replace("D", "E"))
            power = float(raw_power.replace("d", "e").replace("D", "E"))
        except Exception as e:
            raise ValueError(
                f"从 .mt0 提取数值失败: {e}, "
                f"原始值: delay='{raw_delay}', power='{raw_power}', power_col='{power_col_name}'"
            ) from e

        return delay, power

    async def execute_circuit(self, work_dir: Path, circuit_type: str):
        """在单个子目录下执行 HSPICE。

        返回字典而不是抛异常，这样上层可以在部分电路失败时继续处理其它电路。
        只有环境级致命错误（如找不到 HSPICE 可执行文件）才通过异常向上抛出。
        """
        async with self.semaphore:
            circuit_dir = work_dir / circuit_type
            hspice_bin = os.getenv("HSPICE_BIN") or shutil.which("hspice")
            if not hspice_bin:
                # 这是环境级错误，直接抛出异常，由上层统一处理
                raise RuntimeError("HSPICE not found. Set HSPICE_BIN environment variable.")

            # 日志打印到电路子目录的 sim.log 文件
            log_path = circuit_dir / "sim.log"
            process = await asyncio.create_subprocess_exec(
                hspice_bin,
                "run_hspice.sp",
                cwd=circuit_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "circuit_type": circuit_type,
                    "success": False,
                    "error": f"HSPICE timeout after {self.TIMEOUT} seconds for {circuit_type}",
                }

            stdout_text = stdout.decode("utf-8") if stdout else ""
            stderr_text = stderr.decode("utf-8") if stderr else ""

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(stdout_text + "\n" + stderr_text)

            if process.returncode != 0:
                return {
                    "circuit_type": circuit_type,
                    "success": False,
                    "error": f"HSPICE failed with return code {process.returncode}",
                    "returncode": process.returncode,
                }

            return {
                "circuit_type": circuit_type,
                "success": True,
                "error": "",
                "returncode": process.returncode,
            }
        
