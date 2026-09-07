import asyncio
from typing import Any

from .tcad_tool import TCADTool
from .common import format_error_response, format_success_response


class BatchTCADTool(TCADTool):
    """并发运行多组相互独立的 TCAD 参数候选。"""

    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)

        self.max_candidates = int(
            config.get("max_candidates", 2)
        )

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        candidates = parameters.get("candidates", [])

        if not candidates:
            return (
                format_error_response(
                    "Missing required parameter: candidates"
                ),
                0,
                {},
            )

        if len(candidates) > self.max_candidates:
            return (
                format_error_response(
                    f"At most {self.max_candidates} candidates "
                    "may be submitted in one batch."
                ),
                0,
                {},
            )

        candidate_ids = set()

        for index, candidate in enumerate(candidates):
            candidate_id = candidate.get(
                "candidate_id",
                f"candidate_{index + 1}",
            )

            if candidate_id in candidate_ids:
                return (
                    format_error_response(
                        f"Duplicate candidate_id: {candidate_id}"
                    ),
                    0,
                    {},
                )

            candidate_ids.add(candidate_id)
            candidate["candidate_id"] = candidate_id

        # 保存父类方法，后面并发调用原有 TCADTool.execute()
        run_one_candidate = super().execute

        async def run_candidate(candidate):
            candidate_id = candidate["candidate_id"]

            # candidate_id 只用于批次识别，
            # 原 TCADTool.execute() 不需要这个字段。
            tcad_parameters = {
                key: value
                for key, value in candidate.items()
                if key != "candidate_id"
            }

            try:
                response, reward, metrics = await run_one_candidate(
                    instance_id=instance_id,
                    parameters=tcad_parameters,
                    **kwargs,
                )

                return {
                    "candidate_id": candidate_id,
                    "status": (
                        "success" if reward > 0 else "failed"
                    ),
                    "parameters": tcad_parameters,
                    "reward": reward,
                    "metrics": metrics,
                    "tool_response": response,
                }

            except Exception as exc:
                # 一个候选失败，不取消另一个候选
                return {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "parameters": tcad_parameters,
                    "reward": 0,
                    "metrics": {},
                    "error": str(exc),
                }

        # 创建多个候选任务
        tasks = [
            run_candidate(candidate)
            for candidate in candidates
        ]

        # 汇合点：等待所有候选结束后才向 Agent 返回
        results = await asyncio.gather(
            *tasks,
            return_exceptions=False,
        )

        succeeded = [
            result
            for result in results
            if result["status"] == "success"
        ]

        response_data = {
            "candidate_count": len(results),
            "success_count": len(succeeded),
            "failure_count": len(results) - len(succeeded),
            "candidates": results,
        }

        reward = 1.0 if succeeded else 0.0

        return (
            format_success_response(response_data),
            reward,
            response_data,
        )