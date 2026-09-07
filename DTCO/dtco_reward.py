import json
import math
import re
from typing import Any, Dict, List, Optional

from internbootcamp.src.base_reward_calculator import BaseRewardCalculator
from .tcad_rewards_simple import TCADScoreCalculator


def _extract_tool_response_blocks(response: str) -> List[str]:
    """从完整对话中提取所有 <tool_response>...</tool_response> 块。"""
    if not response:
        return []
    return [
        match.group(1)
        for match in re.finditer(r"<tool_response>(.*?)</tool_response>", response, flags=re.DOTALL)
    ]


def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """尽可能从一段文本中解析出一个 JSON dict（宽松模式）。"""
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    """安全地转换为 float，失败时返回 None。"""
    try:
        return float(value)
    except Exception:
        return None


def _clip01(x: float) -> float:
    """裁剪到 [0, 1] 区间。"""
    return max(0.0, min(1.0, float(x)))


class DTCORewardCalculator(BaseRewardCalculator):
    """
    DTCO reward 计算器（更新版）。

    - 从 TCAD 工具（DTCO_tcad_simulate）的 JSON 输出中读取 n/p-type 器件指标：
      Ion / Ioff / SS / VDD / width_um；
    - 从 HSPICE 工具（DTCO_hspice_sim）的文本输出中解析各电路的 delay / power；
    - 使用 ground_truth 中的 Ion_target / Ioff_target / SS_target 等信息，
      计算 0～1 归一化的 reward，兼顾器件与电路两个层级。
    - 电路子分使用指数衰减 score=exp(-value/scale)，默认 scale 为 50ps、0.5mW，
      与典型 inverter 目标一致；可通过 ground_truth 或 kwargs 传入
      delay_target_s/delay_scale_s、power_target_W/power_scale_W 覆盖。
    """

    # ==========================
    # 解析阶段：从对话中抽取 metrics
    # ==========================

    @staticmethod
    def _parse_tcad_block(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析单个 TCAD tool_response（JSON）为统一结构。"""
        if not isinstance(parsed, dict):
            return None

        # 兼容 both / 单一类型 两种返回结构
        device_type = parsed.get("device_type")
        n_metrics = None
        p_metrics = None

        if device_type == "both":
            n_res = parsed.get("ntype_result") or {}
            p_res = parsed.get("ptype_result") or {}
            n_metrics = n_res.get("metrics") or {}
            p_metrics = p_res.get("metrics") or {}
        elif device_type == "n-type":
            res = parsed.get("result") or {}
            n_metrics = res.get("metrics") or {}
        elif device_type == "p-type":
            res = parsed.get("result") or {}
            p_metrics = res.get("metrics") or {}
        else:
            # 尝试直接在顶层找 metrics
            top_metrics = parsed.get("metrics")
            if isinstance(top_metrics, dict) and top_metrics:
                n_metrics = top_metrics

        if not any([n_metrics, p_metrics]):
            return None

        def _agg_metrics(n_m: Optional[Dict[str, Any]], p_m: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
            ions: List[float] = []
            ioffs: List[float] = []
            sss: List[float] = []

            for m in (n_m, p_m):
                if not isinstance(m, dict):
                    continue
                ion_v = _to_float(m.get("Ion"))
                ioff_v = _to_float(m.get("Ioff"))
                ss_v = _to_float(m.get("SS"))
                if ion_v is not None:
                    ions.append(ion_v)
                if ioff_v is not None:
                    ioffs.append(ioff_v)
                if ss_v is not None:
                    sss.append(ss_v)

            def _mean(xs: List[float]) -> Optional[float]:
                if not xs:
                    return None
                return float(sum(xs) / len(xs))

            return {
                "Ion": _mean(ions),
                "Ioff": _mean(ioffs),
                "SS": _mean(sss),
            }

        agg = _agg_metrics(n_metrics, p_metrics)

        return {
            "device_type": device_type,
            "ntype_metrics": n_metrics,
            "ptype_metrics": p_metrics,
            "agg": agg,
        }

    @staticmethod
    def _parse_hspice_block(text: str) -> Optional[Dict[str, Any]]:
        """
        解析 HspiceTool 文本输出：

        形如：
        电路类型: INV, 延迟: 5.2966318e-11 s, 功耗: 1.9778595e-06 W
        电路类型: NAND, 仿真失败: ...
        电路类型: NOR, 仿真结果解析失败: ...
        """
        if not text:
            return None

        lines = text.strip().splitlines()
        success_pattern = re.compile(
            r"电路类型:\s*(\S+),\s*延迟:\s*([eE0-9\.\+\-]+)\s*s,\s*功耗:\s*([eE0-9\.\+\-]+)\s*W"
        )
        failure_pattern = re.compile(
            r"电路类型:\s*(\S+),\s*(仿真失败|仿真结果解析失败):\s*(.+)"
        )

        circuits: List[Dict[str, Any]] = []
        for line in lines:
            success_match = success_pattern.search(line)
            if success_match:
                circuit_type = success_match.group(1)
                delay = _to_float(success_match.group(2))
                power = _to_float(success_match.group(3))
                if delay is None or power is None:
                    continue
                circuits.append(
                    {
                        "circuit_type": circuit_type,
                        "success": True,
                        "delay": float(delay),
                        "power": float(power),
                    }
                )
                continue

            failure_match = failure_pattern.search(line)
            if failure_match:
                circuits.append(
                    {
                        "circuit_type": failure_match.group(1),
                        "success": False,
                        "error_type": failure_match.group(2),
                        "error": failure_match.group(3).strip(),
                    }
                )

        if not circuits:
            return None

        success_circuits = [c for c in circuits if c.get("success")]
        avg_delay = None
        avg_power = None
        if success_circuits:
            # 为兼容旧分析脚本，avg_delay/avg_power 仍保留为成功电路上的均值；
            # reward 计算将改为逐电路聚合，并把失败电路按 0 分计入。
            avg_delay = float(sum(c["delay"] for c in success_circuits) / len(success_circuits))
            avg_power = float(sum(c["power"] for c in success_circuits) / len(success_circuits))

        return {
            "circuits": circuits,
            "avg_delay": avg_delay,
            "avg_power": avg_power,
            "success_count": len(success_circuits),
            "failure_count": len(circuits) - len(success_circuits),
        }

    @staticmethod
    def extract_output(response: str) -> Dict[str, Any]:
        """
        从完整对话字符串中抽取 DTCO flow 的关键指标。

        返回结构示例：
        {
            "tcad_last": {...},         # 最近一次 TCAD 指标（包含 n/p metrics 与聚合值）
            "tcad_history": [...],
            "circuit_last": {...},      # 最近一次 HSPICE 电路指标（包含 per-circuit 与平均值）
            "circuit_history": [...],
            "Ion": 8.2e-4,             # 方便上层使用的聚合 Ion/Ioff/SS
            "Ioff": 1.8e-8,
            "SS": 72.2,
        }
        """
        tool_blocks = _extract_tool_response_blocks(response)
        tcad_history: List[Dict[str, Any]] = []
        circuit_history: List[Dict[str, Any]] = []

        for block in tool_blocks:
            parsed = _parse_json_from_text(block)
            if isinstance(parsed, dict):
                tcad_entry = DTCORewardCalculator._parse_tcad_block(parsed)
                if tcad_entry is not None:
                    tcad_history.append(tcad_entry)
                    continue

            # 不是 TCAD JSON，则尝试按 HSPICE 文本解析
            hspice_entry = DTCORewardCalculator._parse_hspice_block(block)
            if hspice_entry is not None:
                circuit_history.append(hspice_entry)

        summary: Dict[str, Any] = {
            "tcad_history": tcad_history,
            "circuit_history": circuit_history,
        }

        if tcad_history:
            tcad_last = tcad_history[-1]
            summary["tcad_last"] = tcad_last
            agg = tcad_last.get("agg") or {}
            # 向后兼容旧逻辑中直接访问 Ion/Ioff/SS 的用法
            for k in ("Ion", "Ioff", "SS"):
                if k in agg and agg[k] is not None:
                    summary[k] = agg[k]

        if circuit_history:
            circuit_last = circuit_history[-1]
            summary["circuit_last"] = circuit_last
            summary["circuit_metrics"] = circuit_last

        return summary

    @classmethod
    def extract_outputs(cls, response: str) -> List[Dict[str, Any]]:
        """
        为了兼容 BaseRewardCalculator 的调试习惯，这里简单返回
        所有 <tool_response> 中成功解析出的 JSON 字典列表。
        """
        outputs: List[Dict[str, Any]] = []
        for block in _extract_tool_response_blocks(response):
            parsed = _parse_json_from_text(block)
            if isinstance(parsed, dict):
                outputs.append(parsed)
        return outputs

    # ==========================
    # 打分阶段：0～1 归一化 reward
    # ==========================

    @staticmethod
    def _compute_tcad_score(
        ion: Optional[float],
        ioff: Optional[float],
        ss: Optional[float],
        ion_target: Optional[float],
        ioff_target: Optional[float],
        ss_target: Optional[float],
    ) -> float:
        """
        基于 TCAD bootcamp 中的 TCADScoreCalculator 计算 TCAD 子分数，范围 [0,1]。

        - 这里直接复用了 `tcad_rewards.TCADScoreCalculator` 的打分形状；
        - 对于 DTCO，目前 ground_truth 里通常只给出了 Ion/Ioff/SS 的 target，
          所以容忍度参数（*_tolerate）使用基于 target 的启发式默认值。
        """
        if ion is None or ioff is None or ss is None:
            return 0.0

        # 若 target 缺失，用当前值兜底，避免崩掉
        ion_target = ion_target if ion_target is not None and ion_target > 0 else max(ion, 1e-6)
        ioff_target = ioff_target if ioff_target is not None and ioff_target > 0 else max(ioff, 1e-12)
        ss_target = ss_target if ss_target is not None and ss_target > 0 else max(ss, 1.0)

        # 参考 TCAD bootcamp 的典型设置，构造一组合理的容忍度和参数：
        # - Ion_tolerate: 允许 Ion 下降到 target 的一半左右
        # - Ioff_tolerate: 允许 Ioff 上升到 target 的 10 倍左右
        # - SS_tolerate: 允许 SS 提高 20%
        ion_tolerate = max(ion_target * 0.5, 1e-8)
        ioff_tolerate = max(ioff_target * 10.0, 1e-12)
        ss_tolerate = max(ss_target * 1.2, 1e-3)

        score_calculator = TCADScoreCalculator(
            ion_target=ion_target,
            ioff_target=ioff_target,
            ss_target=ss_target,
            ion_tolerate=ion_tolerate,
            ioff_tolerate=ioff_tolerate,
            ss_tolerate=ss_tolerate,
            r_tolerate=0.2,
            p_ioff=1,
            p_ion=1,
            p_ss=2,
            eps_ion=1e-15,
            eps_ioff=1e-15,
            eps_ss=1e-3,
        )

        try:
            return float(score_calculator.compute_score(float(ion), float(ioff), float(ss)))
        except Exception:
            return 0.0

    # 当 指标值=scale 时该项子分约 0.6；可由 identity/kwargs 覆盖
    DEFAULT_DELAY_SCALE_S = 4.5e-11   # 45 ps
    DEFAULT_POWER_SCALE_W = 2.7e-6    # 2.7 uW

    @classmethod
    def _compute_single_circuit_score(
        cls,
        delay: Optional[float],
        power: Optional[float],
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        if delay is None or power is None:
            return 0.0

        ground_truth = ground_truth or {}
        delay_scale = _to_float(ground_truth.get("delay_scale_s") or kwargs.get("delay_scale_s"))
        power_scale = _to_float(ground_truth.get("power_scale_W") or kwargs.get("power_scale_W"))
        if delay_scale is None or delay_scale <= 0:
            delay_scale = cls.DEFAULT_DELAY_SCALE_S
        if power_scale is None or power_scale <= 0:
            power_scale = cls.DEFAULT_POWER_SCALE_W

        # 非线性：scale 为及格线(score=0.6)，rate*scale 为优秀线(score=1.0)，rate 默认 0.5
        rate = 0.5
        score_delay = math.exp(-max(0.0, float(delay) / float(delay_scale) - rate) / (1.0 - rate) * math.log(1.0 / 0.6))
        score_power = math.exp(-max(0.0, float(power) / float(power_scale) - rate) / (1.0 - rate) * math.log(1.0 / 0.6))

        score_delay = _clip01(score_delay)
        score_power = _clip01(score_power)

        # 几何平均：score_delay * score_power
        final_score = math.sqrt(score_delay * score_power)
        return float(final_score)

    @classmethod
    def _compute_circuit_score(
        cls,
        circuit_last: Optional[Dict[str, Any]],
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        基于 HSPICE 电路结果计算电路子分数，范围 [0,1]。

        - 新格式：逐电路计分后求平均，失败电路按 0 分计入，避免“只对成功子集求均值”。
        - 旧格式：若只有 avg_delay/avg_power，则回退到原有逻辑。
        """
        if not isinstance(circuit_last, dict):
            return 0.0

        circuits = circuit_last.get("circuits")
        if isinstance(circuits, list) and circuits:
            per_circuit_scores: List[float] = []
            for circuit in circuits:
                if not isinstance(circuit, dict):
                    continue
                if not circuit.get("success", False):
                    per_circuit_scores.append(0.0)
                    continue

                delay = _to_float(circuit.get("delay"))
                power = _to_float(circuit.get("power"))
                per_circuit_scores.append(
                    cls._compute_single_circuit_score(delay, power, ground_truth=ground_truth, **kwargs)
                )

            if per_circuit_scores:
                return float(sum(per_circuit_scores) / len(per_circuit_scores))

        avg_delay = _to_float(circuit_last.get("avg_delay"))
        avg_power = _to_float(circuit_last.get("avg_power"))
        return cls._compute_single_circuit_score(avg_delay, avg_power, ground_truth=ground_truth, **kwargs)

    @classmethod
    def _verify_correction(cls, extracted_output: Dict[str, Any], ground_truth: dict, **kwargs) -> float:
        """
        计算最终 reward（策略 B：先算每步总分，再取 first/best/last）。
        - 每一步 = 一次 circuit 仿真（circuit_history 的一条）；
        - 每步总分 = 0.1 * tcad_score + 0.9 * circuit_score_i（该步的 TCAD 使用当前已有的最新 TCAD）；
        - first = 第一步总分，best = 所有步总分最大值，last = 最后一步总分；
        - 最终分 = w_first*first + w_best*best + w_last*last（权重可配，默认 0.2/0.3/0.5）。
        - 若 TCAD 不合格（tcad_score<=0），最终分为 0。
        """
        if not isinstance(extracted_output, dict) or not extracted_output:
            return 0.0

        ground_truth = ground_truth or {}
        tcad_history = extracted_output.get("tcad_history") or []
        circuit_history = extracted_output.get("circuit_history") or []

        # 1) 计算用于各步的 TCAD 分数（使用最新一次 TCAD 结果，与现有语义一致）
        tcad_last = extracted_output.get("tcad_last") or (tcad_history[-1] if tcad_history else None)
        if tcad_last and isinstance(tcad_last, dict):
            agg = tcad_last.get("agg") or {}
            ion = _to_float(agg.get("Ion", extracted_output.get("Ion")))
            ioff = _to_float(agg.get("Ioff", extracted_output.get("Ioff")))
            ss = _to_float(agg.get("SS", extracted_output.get("SS")))
        else:
            ion = ioff = ss = None
        ion_target = _to_float(ground_truth.get("Ion_target"))
        ioff_target = _to_float(ground_truth.get("Ioff_target"))
        ss_target = _to_float(ground_truth.get("SS_target"))
        
        if ion_target is None or ioff_target is None or ss_target is None:
            # 必需目标缺失，直接返回 0
            return 0.0
            
        tcad_score = cls._compute_tcad_score(ion, ioff, ss, ion_target, ioff_target, ss_target)

        if tcad_score <= 0.0:
            return 0.0

        # 2) 无 circuit 仿真时退化为仅 TCAD：单步即 first=best=last=tcad_score
        w_tcad = 0.1
        w_circuit = 0.9
        if not circuit_history:
            return _clip01(w_tcad * tcad_score) 

        # 3) 每一步 = 一次 circuit 仿真，计算每步总分

        step_scores: List[float] = []
        for circuit_entry in circuit_history:
            circ_score = cls._compute_circuit_score(circuit_entry, ground_truth=ground_truth, **kwargs)
            step_score = w_tcad * tcad_score + w_circuit * circ_score
            step_scores.append(_clip01(step_score))

        first_score = step_scores[0]
        best_score = float(max(step_scores))
        last_score = step_scores[-1]

        # 4) first/best/last 权重（可从 ground_truth 或 kwargs 覆盖）
        w_first = _to_float(ground_truth.get("first_weight") or kwargs.get("first_weight")) or 0
        w_best = _to_float(ground_truth.get("best_weight") or kwargs.get("best_weight")) or 0.7
        w_last = _to_float(ground_truth.get("last_weight") or kwargs.get("last_weight")) or 0.3
        # 归一化
        total_w = w_first + w_best + w_last
        if total_w > 0:
            w_first, w_best, w_last = w_first / total_w, w_best / total_w, w_last / total_w

        final_score = w_first * first_score + w_best * best_score + w_last * last_score
        # print(f"first_score: {first_score}, best_score: {best_score}, last_score: {last_score}, final_score: {final_score}")
        return _clip01(final_score)


def compute_dtco_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    eval_mode: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    VERL 兼容的 DTCO reward 入口。

    - 输入：完整对话字符串 solution_str（包含 <tool_response>）、ground_truth 指标；
    - 输出：0～1 的最终 score 以及 TCAD / 电路层面的详细指标。
    """
    _ = data_source
    _ = eval_mode
    extra_info = extra_info or {}

    calculator = DTCORewardCalculator()
    extracted = calculator.extract_output(solution_str)
    score = calculator._verify_correction(extracted, ground_truth or {}, **kwargs)

    # 保留 num_turns 语义：等于 tool_response 的数量
    outputs = calculator.extract_outputs(solution_str)

    ion = _to_float(extracted.get("Ion")) if isinstance(extracted, dict) else None
    ioff = _to_float(extracted.get("Ioff")) if isinstance(extracted, dict) else None
    ss = _to_float(extracted.get("SS")) if isinstance(extracted, dict) else None

    result: Dict[str, Any] = {
        "score": float(score),
        "ion": ion,
        "ioff": ioff,
        "ss": ss,
        "is_valid": ion is not None and ioff is not None and ss is not None,
        "num_turns": len(outputs),
    }

    # 附加 TCAD / 电路详细信息，方便后处理分析
    if isinstance(extracted, dict):
        if "tcad_last" in extracted:
            result["tcad_metrics"] = extracted["tcad_last"]
        if "circuit_metrics" in extracted:
            result["circuit_metrics"] = extracted["circuit_metrics"]

    return result


if __name__ == "__main__":
    # 测试 first/best/last 策略 B：3 步，每步都有 TCAD + circuit 仿真（多电路类型）
    def tcad_block(n_ion=8e-4, n_ioff=2e-8, n_ss=70.0, p_ion=6e-4, p_ioff=1.5e-8, p_ss=72.0):
        return json.dumps({
            "device_type": "both",
            "ntype_result": {"metrics": {"Ion": n_ion, "Ioff": n_ioff, "SS": n_ss}},
            "ptype_result": {"metrics": {"Ion": p_ion, "Ioff": p_ioff, "SS": p_ss}},
        })
    # 每步 = 一次 TCAD + 一次完整 circuit 仿真（INV + NAND + NOR）
    step1_tcad = tcad_block()
    step1_circuit = (
        "电路类型: INV, 延迟: 6.0e-11 s, 功耗: 2.5e-03 W\n"
    )
    step2_tcad = tcad_block()
    step2_circuit = (
        "电路类型: INV, 延迟: 4.5e-12 s, 功耗: 2.7e-07 W\n"
    )
    step3_tcad = tcad_block()
    step3_circuit = (
        "电路类型: INV, 延迟: 4.5e-11 s, 功耗: 2.7e-06 W\n"
    )
    solution_str = (
        "<tool_response>\n" + step1_tcad + "\n</tool_response>\n"
    )
    identity = {
        "Ion_target": 8e-4,
        "Ioff_target": 2e-8,
        "SS_target": 72.0,
    }
    result = compute_dtco_reward(
        data_source="test",
        solution_str=solution_str,
        ground_truth=identity,
    )
    print("compute_dtco_reward test (3 steps, each with TCAD + circuit):")
    print(f"  score = {result['score']:.4f}")
    print(f"  is_valid = {result['is_valid']}, num_turns = {result['num_turns']}")
    assert 0 <= result["score"] <= 1, "score 应在 [0,1]"
    print("  OK")

