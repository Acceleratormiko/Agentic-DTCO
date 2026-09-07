

import asyncio
import gc
import hashlib
import json
import os
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from internbootcamp.src.base_tool import BaseTool
from internbootcamp.src.openai_tool_schema import OpenAIFunctionToolSchema
# from internbootcamp.common.compat import rollout_trace_op

from .tcad_to_dataset import build_nn_dataset


class CompactModelTool(BaseTool):
    """
    生成 compact model（训练并导出 Verilog-A），整体流程分为 3 步：

    0) 从 config.base_dir/{ntype,ptype} 复制：
       - IV_nn_data_des.cmd
       - CV_nn_data_des.cmd
       到 work_dir/{ntype,ptype}/ 下
       然后在各自目录运行：
       - sdevice --exit-on-failure IV_nn_data_des.cmd
       - sdevice --exit-on-failure CV_nn_data_des.cmd
       以生成 result_*.plt / result_*_ac_des.plt（即 main.py 里列出的那些 plt）

    1) tcad_to_dataset.build_nn_dataset: 读取上述 plt -> 4个CSV
    2) csv_to_va.dataset_to_va: 训练并生成 nnmodel_{n,p}_type.va

    目录约定：
    - work_dir/
        ├── ntype/
        └── ptype/
    - base_dir/ (模板目录，必须存在)
        ├── ntype/IV_nn_data_des.cmd, CV_nn_data_des.cmd
        └── ptype/IV_nn_data_des.cmd, CV_nn_data_des.cmd
    """

    _IV_CMD = "IV_nn_data_des.cmd"
    _CV_CMD = "CV_nn_data_des.cmd"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        """原子写状态文件，避免监控脚本读到半截JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
        tmp_path.replace(path)

    def _publish_training_status(
        self,
        status_paths: list[Path],
        payload: Dict[str, Any],
    ) -> None:
        """写工作目录状态，并尽力写入全局监控注册目录。"""
        for index, path in enumerate(status_paths):
            try:
                self._write_json_atomic(path, payload)
            except Exception:
                # work_dir/models中的第一个状态文件属于任务产物，写失败应暴露；
                # /tmp注册副本只是监控便利项，失败不能导致训练失败。
                if index == 0:
                    raise

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        assert config is not None, "CompactModelTool __init__需要提供config参数"
        assert tool_schema is not None, "CompactModelTool __init__需要提供tool_schema参数"
        super().__init__(config, tool_schema)

        semaphore = config.get("semaphore", 1)
        self.semaphore = asyncio.Semaphore(semaphore)

        base_dir = config.get("base_dir",'internbootcamp/bootcamps/DTCO/configs/nnmodel_cmd')
        if not base_dir:
            raise ValueError("CompactModelTool __init__失败，config参数缺少base_dir键")
        self.base_dir = Path(base_dir).resolve()
        self._validate_template_dir(self.base_dir)

        self.csv_filenames = config.get(
            "csv_filenames", ["n_iv.csv", "p_iv.csv", "n_cv.csv", "p_cv.csv"]
        )
        self.models_subdir = "models"

    def _validate_template_dir(self, base_dir: Path) -> None:
        n = base_dir / "ntype"
        p = base_dir / "ptype"
        for d in (n, p):
            if not d.is_dir():
                raise FileNotFoundError(f"base_dir 缺少目录: {d}")
            for cmd in (self._IV_CMD, self._CV_CMD):
                if not (d / cmd).is_file():
                    raise FileNotFoundError(f"base_dir 缺少模板文件: {(d / cmd)}")

    def _validate_work_dir(self, work_dir: Path) -> Tuple[Path, Path]:
        if not work_dir.exists() or not work_dir.is_dir():
            raise FileNotFoundError(f"work_dir 不存在或不是目录: {work_dir}")
        ntype_dir = work_dir / "ntype"
        ptype_dir = work_dir / "ptype"
        if not ntype_dir.is_dir():
            raise FileNotFoundError(f"work_dir 缺少 ntype 子目录: {ntype_dir}")
        if not ptype_dir.is_dir():
            raise FileNotFoundError(f"work_dir 缺少 ptype 子目录: {ptype_dir}")
        return ntype_dir, ptype_dir

    def _find_mesh_file(self, device_dir: Path) -> str:
        """
        Sentaurus 网格文件：
        - 优先使用 tcad_tool 的默认命名 mos_structure_msh.tdr
        - 否则兜底查找 *_msh.tdr
        """
        preferred = device_dir / "mos_structure_msh.tdr"
        if preferred.is_file():
            return preferred.name
        matches = sorted(device_dir.glob("*_msh.tdr"))
        if matches:
            return matches[0].name
        raise FileNotFoundError(f"未找到网格文件 *_msh.tdr（或 mos_structure_msh.tdr）于: {device_dir}")

    def _patch_grid_in_cmd(self, cmd_path: Path, mesh_filename: str) -> None:
        """
        将 cmd 里 Grid="sde_result_msh.tdr" 改为实际 mesh 文件名。
        """
        txt = cmd_path.read_text(encoding="utf-8", errors="ignore")
        # 1) 通用替换 sde_result_msh.tdr 字面量
        txt2 = txt.replace('Grid      = "sde_result_msh.tdr"', f'Grid      = "{mesh_filename}"')
        txt2 = txt2.replace('Grid = "sde_result_msh.tdr"', f'Grid = "{mesh_filename}"')
        # 2) 兜底：正则替换任意 Grid="xxx"
        txt2 = re.sub(r'(\bGrid\s*=\s*")([^"]+)(")', rf'\1{mesh_filename}\3', txt2)
        if txt2 != txt:
            cmd_path.write_text(txt2, encoding="utf-8")

    def _copy_and_prepare_cmds(self, device_dir: Path, template_dir: Path) -> Dict[str, str]:
        """
        复制 IV/CV cmd 到 device_dir，并修正 Grid 指向正确 mesh。
        返回复制后的 cmd 路径（字符串）。
        """
        mesh_filename = self._find_mesh_file(device_dir)

        iv_src = template_dir / self._IV_CMD
        cv_src = template_dir / self._CV_CMD
        iv_dst = device_dir / self._IV_CMD
        cv_dst = device_dir / self._CV_CMD

        shutil.copyfile(iv_src, iv_dst)
        shutil.copyfile(cv_src, cv_dst)

        self._patch_grid_in_cmd(iv_dst, mesh_filename)
        self._patch_grid_in_cmd(cv_dst, mesh_filename)

        return {"iv_cmd": str(iv_dst), "cv_cmd": str(cv_dst), "mesh": mesh_filename}

    async def _run_sdevice(self, cwd: Path, cmd_filename: str) -> int:
        if shutil.which("sdevice") is None:
            raise FileNotFoundError("找不到可执行文件 'sdevice'（请确认 Sentaurus 环境已配置到 PATH）")

        proc = await asyncio.create_subprocess_exec(
            "sdevice",
            "--exit-on-failure",
            cmd_filename,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # 消费输出，避免 buffer 堵塞
        if proc.stdout is not None:
            async for _ in proc.stdout:
                pass
        ret = await proc.wait()
        if ret != 0:
            raise RuntimeError(f"sdevice 运行失败: cwd={cwd}, cmd={cmd_filename}, ret={ret}")
        return ret

    def _collect_iv_plts(self, device_dir: Path) -> list[str]:
        """
        对应 IV_nn_data_des.cmd 的输出：result_Vg_*_IdVd_des.plt（大小写不敏感）。
        """
        plts = [p for p in device_dir.glob("*.plt") if p.is_file()]
        preferred = [p for p in plts if re.search(r"^result_.*idvd_des\.plt$", p.name, flags=re.IGNORECASE)]
        candidates = preferred if preferred else plts
        return [str(p.resolve()) for p in sorted(candidates, key=lambda x: x.name)]

    def _collect_cv_plts(self, device_dir: Path) -> list[str]:
        """
        对应 CV_nn_data_des.cmd 的输出：result_vg_*_ac_des.plt（大小写不敏感）。
        """
        plts = [p for p in device_dir.glob("*.plt") if p.is_file()]
        preferred = [p for p in plts if re.search(r"^result_.*_ac_des\.plt$", p.name, flags=re.IGNORECASE)]
        candidates = preferred if preferred else plts
        return [str(p.resolve()) for p in sorted(candidates, key=lambda x: x.name)]

    # @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """
        parameters:
          - work_dir: tcad_tool 输出的 work dir（包含 ntype/ ptype/）

        说明：
        - 生成的 CSV / 模型 / VA 均写入 work_dir/ntype 与 work_dir/ptype 内部
        - sdevice 执行：ntype 与 ptype 两条链路并行；每条链路内部按 IV -> CV 顺序执行。
        """
        training_log_path: Path | None = None
        training_status_paths: list[Path] = []
        training_status: Dict[str, Any] = {}

        async with self.semaphore:
            try:
                work_dir = parameters.get("work_dir")
                if not work_dir:
                    return "执行CompactModelTool失败，缺少参数 work_dir", 0, {}
                work_dir = Path(work_dir)

                ntype_dir, ptype_dir = self._validate_work_dir(work_dir)

                n_prep = self._copy_and_prepare_cmds(ntype_dir, self.base_dir / "ntype")
                p_prep = self._copy_and_prepare_cmds(ptype_dir, self.base_dir / "ptype")

                # 创建 sdevice 任务列表（参考 tcad_tool.py 的并行风格）
                sdevice_task_list = []
                sdevice_task_list.append(self._run_sdevice(ntype_dir, self._IV_CMD))
                sdevice_task_list.append(self._run_sdevice(ntype_dir, self._CV_CMD))
                sdevice_task_list.append(self._run_sdevice(ptype_dir, self._IV_CMD))
                sdevice_task_list.append(self._run_sdevice(ptype_dir, self._CV_CMD))

                # 并行执行所有 sdevice 仿真
                results = await asyncio.gather(*sdevice_task_list, return_exceptions=True)
                
                # 检查是否有任务失败
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        raise RuntimeError(f"sdevice 任务 {i} 失败: {res}")

                plt_n_iv_list = self._collect_iv_plts(ntype_dir)
                plt_p_iv_list = self._collect_iv_plts(ptype_dir)
                plt_n_cv_list = self._collect_cv_plts(ntype_dir)
                plt_p_cv_list = self._collect_cv_plts(ptype_dir)

                if not plt_n_iv_list or not plt_p_iv_list:
                    raise FileNotFoundError("未生成 IV .plt（请检查 sdevice 输出与 cmd 配置）")
                if not plt_n_cv_list or not plt_p_cv_list:
                    raise FileNotFoundError("未生成 CV .plt（请检查 sdevice 输出与 cmd 配置）")

                if len(self.csv_filenames) != 4:
                    raise ValueError("config.csv_filenames 必须是长度为4的列表: [n_iv, p_iv, n_cv, p_cv]")

                # 1) 生成 CSV：分别落到 ntype/ 与 ptype/ 下
                # 注意：build_nn_dataset 一次生成四个 CSV，因此这里把 n_xxx 输出到 ntype，p_xxx 输出到 ptype
                csv_paths = [
                    str((ntype_dir / self.csv_filenames[0]).resolve()),  # n_iv.csv
                    str((ptype_dir / self.csv_filenames[1]).resolve()),  # p_iv.csv
                    str((ntype_dir / self.csv_filenames[2]).resolve()),  # n_cv.csv
                    str((ptype_dir / self.csv_filenames[3]).resolve()),  # p_cv.csv
                ]
                n_iv_csv, p_iv_csv, n_cv_csv, p_cv_csv = build_nn_dataset(
                    plt_n_iv_list,
                    plt_p_iv_list,
                    plt_n_cv_list,
                    plt_p_cv_list,
                    filelist=csv_paths,
                )

                # 2) 训练并导出 VA
                # dataset_to_va 需要同时看到四个 CSV，这里先统一输出到 work_dir/models，
                models_dir = work_dir / self.models_subdir
                models_dir.mkdir(parents=True, exist_ok=True)

                from . import csv_to_va

                # Agent CLI本身使用动态终端重绘；tqdm/逐epoch输出若继续写主终端，
                # 会与CLI争用光标。训练输出统一写入独立日志，主界面只保留工具运行状态。
                training_log_path = models_dir / "nn_training.log"
                local_status_path = models_dir / "nn_training_status.json"
                monitor_dir = Path(
                    os.environ.get(
                        "DTCO_TRAINING_MONITOR_DIR",
                        "/tmp/agentic_dtco_training",
                    )
                )
                status_key = hashlib.sha1(
                    str(work_dir.resolve()).encode("utf-8")
                ).hexdigest()[:12]
                registry_status_path = monitor_dir / f"{work_dir.name}_{status_key}.json"
                training_status_paths = [local_status_path, registry_status_path]

                training_status = {
                    "schema_version": "nn-training-status.v1",
                    "state": "TRAINING",
                    "pid": os.getpid(),
                    "instance_id": instance_id,
                    "work_dir": str(work_dir.resolve()),
                    "models_dir": str(models_dir.resolve()),
                    "log_file": str(training_log_path.resolve()),
                    "started_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                    "finished_at": None,
                    "error": "",
                }
                self._publish_training_status(training_status_paths, training_status)

                print(
                    f"[INFO] NN模型正在训练；详细进度已写入: {training_log_path.resolve()}",
                    flush=True,
                )

                with training_log_path.open(
                    "w",
                    encoding="utf-8",
                    buffering=1,
                ) as training_log:
                    training_log.write(
                        f"[{self._now_iso()}] NN training started\n"
                        f"work_dir={work_dir.resolve()}\n"
                        f"models_dir={models_dir.resolve()}\n"
                    )
                    training_log.flush()
                    csv_to_va.set_training_output(training_log)
                    try:
                        p_va_path, n_va_path = csv_to_va.dataset_to_va(
                            n_iv_csv=n_iv_csv,
                            p_iv_csv=p_iv_csv,
                            n_cv_csv=n_cv_csv,
                            p_cv_csv=p_cv_csv,
                            models_dir=str(models_dir),
                        )
                    finally:
                        csv_to_va.set_training_output(None)
                    training_log.write(
                        f"[{self._now_iso()}] NN training and Verilog-A export completed\n"
                    )
                    training_log.flush()

                training_status.update(
                    {
                        "state": "COMPLETED",
                        "updated_at": self._now_iso(),
                        "finished_at": self._now_iso(),
                    }
                )
                self._publish_training_status(training_status_paths, training_status)
                print(
                    f"[INFO] NN模型训练与Verilog-A导出完成: {models_dir.resolve()}",
                    flush=True,
                )

                # 训练和导出完成后显式释放内存，避免多次调用时累积
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

                # VA 与所有训练产物统一保存在 work_dir/models 下

                response_data: Dict[str, Any] = {
                    "work_dir": str(work_dir),
                    "csv": {
                        "n_iv_csv": n_iv_csv,
                        "p_iv_csv": p_iv_csv,
                        "n_cv_csv": n_cv_csv,
                        "p_cv_csv": p_cv_csv,
                    },
                    "models_dir": str(models_dir),
                    "va": {
                        "n_va_path": n_va_path,
                        "p_va_path": p_va_path,
                    },
                    "training": {
                        "state": "COMPLETED",
                        "log_path": str(training_log_path.resolve()),
                        "status_path": str(training_status_paths[0].resolve()),
                    },
                }

                try:
                    out_json = work_dir / "compact_model_result.json"
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(response_data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

                return json.dumps(response_data, ensure_ascii=False), 1.0, response_data

            except Exception as e:
                tb = traceback.format_exc()
                if training_log_path is not None:
                    try:
                        with training_log_path.open("a", encoding="utf-8") as training_log:
                            training_log.write(
                                f"[{self._now_iso()}] NN training failed: {e}\n"
                            )
                            training_log.flush()
                    except Exception:
                        pass
                if training_status_paths:
                    try:
                        training_status.update(
                            {
                                "state": "FAILED",
                                "updated_at": self._now_iso(),
                                "finished_at": self._now_iso(),
                                "error": str(e),
                            }
                        )
                        self._publish_training_status(
                            training_status_paths,
                            training_status,
                        )
                    except Exception:
                        pass
                err = {
                    "error": str(e),
                    "traceback": tb,
                }
                return json.dumps(err, ensure_ascii=False), 0, {}
