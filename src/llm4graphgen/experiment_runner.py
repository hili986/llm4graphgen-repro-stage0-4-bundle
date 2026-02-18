"""实验批量运行器 — 自动化运行全部实验，支持并行、断点续跑、错误检测。

功能：
1. 预定义实验计划（minimal / standard / full）
2. 可配置并行度（默认 2，兼顾 API 速率限制）
3. 自动检测 API 额度不足、网络错误等，暂停并报告
4. 断点续跑：已完成的实验自动跳过
5. 实时状态报告

使用方式：
    # 查看实验计划（不运行）
    python -m llm4graphgen.experiment_runner --plan standard --dry-run

    # 运行标准实验计划（2 并行）
    python -m llm4graphgen.experiment_runner --plan standard --parallel 2

    # 运行完整实验计划（含 Small/Large 消融）
    python -m llm4graphgen.experiment_runner --plan full --parallel 2

    # 断点续跑（自动跳过已完成的）
    python -m llm4graphgen.experiment_runner --plan standard --parallel 2

    # 只运行 Stage4
    python -m llm4graphgen.experiment_runner --plan standard --parallel 2 --stage stage4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


# ---------------------------------------------------------------------------
# 实验状态
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"       # 用户手动跳过
    RATE_LIMITED = "rate_limited"  # API 额度不足，需要等待


@dataclass
class ExperimentConfig:
    """单个实验的配置。"""
    name: str                  # 唯一标识，如 "stage2_gpt-4_zero_shot_medium"
    stage: str                 # "stage2" / "stage3" / "stage4"
    module: str                # "llm4graphgen.stage2_rule_based" 等
    args: list[str]            # CLI 参数列表
    description: str = ""      # 人类可读描述

    def cli_command(self) -> list[str]:
        return [sys.executable, "-X", "utf8", "-m", self.module] + self.args


@dataclass
class ExperimentResult:
    """单个实验的运行结果。"""
    name: str
    status: str = Status.PENDING
    start_time: str | None = None
    end_time: str | None = None
    duration_sec: float = 0.0
    return_code: int | None = None
    error_type: str | None = None   # "rate_limit" / "auth" / "network" / "unknown"
    error_detail: str | None = None
    output_dir: str | None = None
    log_tail: str = ""              # 最后几行输出


# ---------------------------------------------------------------------------
# 错误检测
# ---------------------------------------------------------------------------

# API 错误关键词 → 错误类型映射
_ERROR_PATTERNS: list[tuple[str, str]] = [
    # 速率限制
    ("rate limit", "rate_limit"),
    ("rate_limit", "rate_limit"),
    ("429", "rate_limit"),
    ("too many requests", "rate_limit"),
    ("quota exceeded", "rate_limit"),
    ("insufficient_quota", "rate_limit"),
    ("exceeded your current quota", "rate_limit"),
    # 认证错误
    ("401", "auth"),
    ("invalid api key", "auth"),
    ("authentication", "auth"),
    ("unauthorized", "auth"),
    ("incorrect api key", "auth"),
    # 网络错误
    ("connection", "network"),
    ("timeout", "network"),
    ("urlopen error", "network"),
    ("network", "network"),
    ("dns", "network"),
    # 模型不存在
    ("model not found", "model_error"),
    ("does not exist", "model_error"),
]


def detect_error_type(output: str) -> str | None:
    """从输出文本中检测错误类型。"""
    lower = output.lower()
    for pattern, error_type in _ERROR_PATTERNS:
        if pattern in lower:
            return error_type
    return None


def is_fatal_error(error_type: str | None) -> bool:
    """判断是否为致命错误（应停止所有实验）。"""
    return error_type in ("rate_limit", "auth")


# ---------------------------------------------------------------------------
# 实验计划
# ---------------------------------------------------------------------------

def _make_stage2_experiments(model: str, strategies: list[str], sizes: list[str],
                             num_samples: int, num_repeats: int) -> list[ExperimentConfig]:
    """生成 Stage2 实验列表。"""
    exps = []
    for size in sizes:
        for strategy in strategies:
            name = f"stage2_{model}_{strategy}_{size}_{num_samples}s"
            if num_repeats > 1:
                name += f"_r{num_repeats}"
            args = [
                "--provider", "openai",
                "--model", model,
                "--temperature", "0.8",
                "--strategy", strategy,
                "--num-samples", str(num_samples),
                "--size", size,
                "--unique-method", "isomorphism",
            ]
            if num_repeats > 1:
                args += ["--num-repeats", str(num_repeats)]
            exps.append(ExperimentConfig(
                name=name,
                stage="stage2",
                module="llm4graphgen.stage2_rule_based",
                args=args,
                description=f"Stage2 {model} {strategy} {size} ×{num_samples} "
                            f"{'r'+str(num_repeats) if num_repeats > 1 else ''}",
            ))
    return exps


def _make_stage3_experiments(model: str, strategy: str, p_values: str,
                             num_repeats: int) -> list[ExperimentConfig]:
    """生成 Stage3 实验列表。"""
    name = f"stage3_{model}_{strategy}_p{p_values.replace(',', '-')}"
    if num_repeats > 1:
        name += f"_r{num_repeats}"
    args = [
        "--provider", "openai",
        "--model", model,
        "--temperature", "0.5",
        "--strategy", strategy,
        "--p-values", p_values,
    ]
    if num_repeats > 1:
        args += ["--num-repeats", str(num_repeats)]
    return [ExperimentConfig(
        name=name,
        stage="stage3",
        module="llm4graphgen.stage3_distribution",
        args=args,
        description=f"Stage3 {model} {strategy} p=[{p_values}] "
                    f"{'r'+str(num_repeats) if num_repeats > 1 else ''}",
    )]


def _make_stage4_experiments(model: str, classifier: str,
                             paper_tpr_fpr: bool, num_generate: int,
                             strategy: str = "few_shot_cot") -> list[ExperimentConfig]:
    """生成 Stage4 实验列表。"""
    name = f"stage4_{model}_{strategy}_{classifier}"
    if paper_tpr_fpr:
        name += "_paper-tpr"
    name += f"_{num_generate}g"
    args = [
        "--provider", "openai",
        "--model", model,
        "--temperature", "0.5",
        "--strategy", strategy,
        "--num-generate", str(num_generate),
        "--classifier", classifier,
    ]
    if paper_tpr_fpr:
        args.append("--paper-tpr-fpr")
    return [ExperimentConfig(
        name=name,
        stage="stage4",
        module="llm4graphgen.stage4_property",
        args=args,
        description=f"Stage4 {model} {strategy} {classifier} "
                    f"{'paper-tpr ' if paper_tpr_fpr else ''}{num_generate}g",
    )]


def build_plan(plan_name: str, model: str = "gpt-4") -> list[ExperimentConfig]:
    """根据计划名称生成实验列表。

    计划：
    - minimal: 最小验证（1 策略 × medium，快速跑通）
    - standard: 标准复现（4 策略 × medium，对齐论文核心结果）
    - full: 完整消融（4 策略 × 3 规模 + 多模型对比）
    """
    all_strategies = ["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"]

    if plan_name == "minimal":
        exps = []
        exps += _make_stage2_experiments(model, ["zero_shot"], ["medium"], 100, 1)
        exps += _make_stage3_experiments(model, "zero_shot_cot", "0.5", 1)
        exps += _make_stage4_experiments(model, "gin", True, 100)
        return exps

    elif plan_name == "standard":
        exps = []
        # Stage2: 4 策略 × medium × 3 repeats
        exps += _make_stage2_experiments(model, all_strategies, ["medium"], 100, 3)
        # Stage3: all 策略 × 3 repeats
        exps += _make_stage3_experiments(model, "all", "0.2,0.4,0.6,0.8", 3)
        # Stage4: GIN + paper TPR/FPR (推荐) + GIN 自身 TPR/FPR (对照)
        exps += _make_stage4_experiments(model, "gin", True, 100)
        exps += _make_stage4_experiments(model, "gin", False, 100)
        return exps

    elif plan_name == "full":
        exps = []
        # Stage2: 4 策略 × 3 规模 × 3 repeats
        exps += _make_stage2_experiments(model, all_strategies, ["small", "medium", "large"], 100, 3)
        # Stage3: all 策略 × 3 repeats
        exps += _make_stage3_experiments(model, "all", "0.2,0.4,0.6,0.8", 3)
        # Stage4: GIN 两组
        exps += _make_stage4_experiments(model, "gin", True, 100)
        exps += _make_stage4_experiments(model, "gin", False, 100)
        # GPT-3.5-Turbo 对比 (Stage2 medium only)
        exps += _make_stage2_experiments("gpt-3.5-turbo", all_strategies, ["medium"], 100, 3)
        return exps

    else:
        raise ValueError(f"未知计划: {plan_name}，可选: minimal / standard / full")


# ---------------------------------------------------------------------------
# 状态持久化
# ---------------------------------------------------------------------------

_STATUS_FILE = "experiment_status.json"


def _status_path(output_root: str) -> Path:
    return Path(output_root) / _STATUS_FILE


def load_status(output_root: str) -> dict[str, ExperimentResult]:
    """加载实验状态。"""
    path = _status_path(output_root)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for name, d in data.items():
        results[name] = ExperimentResult(**d)
    return results


def save_status(output_root: str, results: dict[str, ExperimentResult]) -> None:
    """保存实验状态。"""
    path = _status_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: asdict(r) for name, r in results.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 单实验执行
# ---------------------------------------------------------------------------

def run_single_experiment(
    exp: ExperimentConfig,
    output_root: str,
    results: dict[str, ExperimentResult],
    lock: threading.Lock,
    stop_event: threading.Event,
) -> ExperimentResult:
    """运行单个实验。"""
    result = ExperimentResult(name=exp.name)
    result.start_time = datetime.now().isoformat(timespec="seconds")
    result.status = Status.RUNNING

    with lock:
        results[exp.name] = result
        save_status(output_root, results)

    if stop_event.is_set():
        result.status = Status.SKIPPED
        result.error_detail = "全局停止信号，跳过此实验"
        result.end_time = datetime.now().isoformat(timespec="seconds")
        with lock:
            results[exp.name] = result
            save_status(output_root, results)
        return result

    cmd = exp.cli_command() + ["--output-root", output_root]
    cmd_str = " ".join(cmd)
    print(f"\n{'='*60}")
    print(f"[启动] {exp.name}")
    print(f"  命令: {cmd_str}")
    print(f"  时间: {result.start_time}")
    print(f"{'='*60}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        output_lines: list[str] = []
        # 实时读取输出
        for line in iter(proc.stdout.readline, ""):
            if stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                result.status = Status.SKIPPED
                result.error_detail = "全局停止信号，实验被终止"
                break

            line = line.rstrip("\n")
            output_lines.append(line)
            print(f"  [{exp.name}] {line}")

            # 实时检测致命错误
            error_type = detect_error_type(line)
            if is_fatal_error(error_type):
                result.error_type = error_type
                result.error_detail = line
                print(f"\n[!!! 致命错误] {exp.name}: {error_type}")
                print(f"  详情: {line}")
                stop_event.set()  # 通知其他线程停止

        proc.wait()
        result.return_code = proc.returncode

        if result.status != Status.SKIPPED:
            if proc.returncode == 0:
                result.status = Status.COMPLETED
            else:
                result.status = Status.FAILED
                # 分析错误类型
                full_output = "\n".join(output_lines[-20:])
                if result.error_type is None:
                    result.error_type = detect_error_type(full_output) or "unknown"
                result.error_detail = output_lines[-1] if output_lines else "无输出"

                if is_fatal_error(result.error_type):
                    stop_event.set()

        result.log_tail = "\n".join(output_lines[-5:])

    except Exception as exc:
        result.status = Status.FAILED
        result.error_type = "exception"
        result.error_detail = str(exc)
        result.log_tail = str(exc)

    result.end_time = datetime.now().isoformat(timespec="seconds")
    if result.start_time and result.end_time:
        t0 = datetime.fromisoformat(result.start_time)
        t1 = datetime.fromisoformat(result.end_time)
        result.duration_sec = (t1 - t0).total_seconds()

    with lock:
        results[exp.name] = result
        save_status(output_root, results)

    status_icon = {
        Status.COMPLETED: "+",
        Status.FAILED: "X",
        Status.SKIPPED: "-",
        Status.RATE_LIMITED: "!",
    }.get(result.status, "?")

    print(f"\n[{status_icon}] {exp.name}: {result.status} "
          f"({result.duration_sec:.0f}s)")
    if result.error_detail and result.status != Status.COMPLETED:
        print(f"    原因: {result.error_detail}")

    return result


# ---------------------------------------------------------------------------
# 主调度器
# ---------------------------------------------------------------------------

def run_experiments(
    plan_name: str,
    output_root: str = "runs",
    parallel: int = 2,
    model: str = "gpt-4",
    stage_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, ExperimentResult]:
    """运行实验计划。"""

    experiments = build_plan(plan_name, model=model)

    # 按 stage 过滤
    if stage_filter:
        experiments = [e for e in experiments if e.stage == stage_filter]

    if not experiments:
        print("没有匹配的实验。")
        return {}

    # 加载已有状态
    existing = load_status(output_root)

    # 确定哪些需要运行
    to_run: list[ExperimentConfig] = []
    skipped_count = 0
    for exp in experiments:
        prev = existing.get(exp.name)
        if prev and prev.status == Status.COMPLETED:
            skipped_count += 1
            continue
        to_run.append(exp)

    # 报告计划
    print("=" * 60)
    print(f"实验计划: {plan_name}")
    print(f"模型: {model}")
    print(f"并行度: {parallel}")
    print(f"输出目录: {output_root}/")
    print(f"总实验数: {len(experiments)}")
    print(f"已完成 (跳过): {skipped_count}")
    print(f"待运行: {len(to_run)}")
    print("=" * 60)

    if to_run:
        print("\n待运行实验列表:")
        for i, exp in enumerate(to_run, 1):
            print(f"  {i:2d}. [{exp.stage}] {exp.name}")
            print(f"      {exp.description}")
    else:
        print("\n所有实验已完成！")

    if skipped_count > 0:
        print(f"\n已跳过 {skipped_count} 个已完成的实验（断点续跑）")

    if dry_run:
        print("\n[预览模式] 以上实验不会实际运行。去掉 --dry-run 开始执行。")
        return existing

    if not to_run:
        return existing

    # 检查环境
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n[错误] 未设置 OPENAI_API_KEY 环境变量！")
        print("  Windows PowerShell: $env:OPENAI_API_KEY = \"sk-xxx\"")
        print("  Linux/macOS:        export OPENAI_API_KEY=\"sk-xxx\"")
        return existing

    print(f"\n{'='*60}")
    print(f"开始执行 {len(to_run)} 个实验 (并行度={parallel})")
    print(f"按 Ctrl+C 可安全中断，下次运行自动续跑")
    print(f"{'='*60}\n")

    results = dict(existing)
    lock = threading.Lock()
    stop_event = threading.Event()
    global_start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_single_experiment, exp, output_root, results, lock, stop_event
                ): exp
                for exp in to_run
            }

            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"\n[异常] {exp.name}: {exc}")
                    with lock:
                        results[exp.name] = ExperimentResult(
                            name=exp.name,
                            status=Status.FAILED,
                            error_type="exception",
                            error_detail=str(exc),
                            end_time=datetime.now().isoformat(timespec="seconds"),
                        )
                        save_status(output_root, results)

                if stop_event.is_set():
                    # 不提交新任务（已提交的会在 run_single_experiment 中检测 stop_event）
                    pass

    except KeyboardInterrupt:
        print("\n\n[中断] 收到 Ctrl+C，正在安全停止...")
        stop_event.set()
        # 标记剩余实验
        with lock:
            for exp in to_run:
                if exp.name in results and results[exp.name].status == Status.RUNNING:
                    results[exp.name].status = Status.FAILED
                    results[exp.name].error_type = "interrupted"
                    results[exp.name].error_detail = "用户中断 (Ctrl+C)"
                elif exp.name not in results or results[exp.name].status == Status.PENDING:
                    results[exp.name] = ExperimentResult(
                        name=exp.name, status=Status.PENDING
                    )
            save_status(output_root, results)

    elapsed = time.time() - global_start

    # 最终报告
    _print_summary(results, experiments, elapsed, output_root)

    return results


def _print_summary(
    results: dict[str, ExperimentResult],
    experiments: list[ExperimentConfig],
    elapsed: float,
    output_root: str,
) -> None:
    """打印运行总结。"""
    print(f"\n{'='*60}")
    print("实验运行总结")
    print(f"{'='*60}")

    completed = sum(1 for r in results.values() if r.status == Status.COMPLETED)
    failed = sum(1 for r in results.values() if r.status == Status.FAILED)
    pending = sum(
        1 for e in experiments
        if e.name not in results or results[e.name].status == Status.PENDING
    )
    rate_limited = sum(1 for r in results.values() if r.error_type == "rate_limit")

    print(f"\n  完成: {completed}")
    print(f"  失败: {failed}")
    print(f"  待运行: {pending}")
    if rate_limited > 0:
        print(f"  API 限额: {rate_limited}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")

    if failed > 0:
        print(f"\n失败的实验:")
        for name, r in results.items():
            if r.status == Status.FAILED:
                print(f"  X {name}")
                print(f"    类型: {r.error_type or 'unknown'}")
                print(f"    详情: {r.error_detail or '无'}")

    if rate_limited > 0:
        print(f"\n[!] 检测到 API 额度限制。建议：")
        print(f"  1. 等待额度恢复后重新运行（已完成的实验会自动跳过）")
        print(f"  2. 检查 OpenAI 账户用量: https://platform.openai.com/usage")
        print(f"  3. 降低并行度: --parallel 1")

    if pending > 0:
        print(f"\n[i] 还有 {pending} 个实验待运行。重新执行相同命令即可续跑。")

    print(f"\n状态文件: {_status_path(output_root)}")
    print(f"结果目录: {output_root}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM4GraphGen 实验批量运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验计划说明:
  minimal   最小验证: 1 策略 × medium, 快速跑通流程 (3 个实验)
  standard  标准复现: 4 策略 × medium × 3 repeats (7 个实验)
  full      完整消融: 4 策略 × 3 规模 × 3 repeats + 多模型 (21 个实验)

示例:
  # 预览计划
  python -m llm4graphgen.experiment_runner --plan standard --dry-run

  # 运行（2 并行，默认 GPT-4）
  python -m llm4graphgen.experiment_runner --plan standard --parallel 2

  # 只运行 Stage2
  python -m llm4graphgen.experiment_runner --plan standard --stage stage2

  # 用 GPT-3.5-Turbo
  python -m llm4graphgen.experiment_runner --plan standard --model gpt-3.5-turbo

  # 断点续跑：直接重新运行同一命令，已完成的会自动跳过
        """,
    )
    parser.add_argument("--plan", default="standard",
                        choices=["minimal", "standard", "full"],
                        help="实验计划 (默认: standard)")
    parser.add_argument("--model", default="gpt-4",
                        help="LLM 模型名 (默认: gpt-4)")
    parser.add_argument("--parallel", type=int, default=2,
                        help="并行实验数 (默认: 2，API 速率限制建议不超过 3)")
    parser.add_argument("--output-root", default="runs",
                        help="输出根目录 (默认: runs)")
    parser.add_argument("--stage", default=None,
                        choices=["stage2", "stage3", "stage4"],
                        help="只运行指定 stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际运行")
    parser.add_argument("--reset", action="store_true",
                        help="清除状态文件，从头开始运行所有实验")
    args = parser.parse_args(argv)

    if args.reset:
        status_path = _status_path(args.output_root)
        if status_path.exists():
            status_path.unlink()
            print(f"已清除状态文件: {status_path}")

    run_experiments(
        plan_name=args.plan,
        output_root=args.output_root,
        parallel=args.parallel,
        model=args.model,
        stage_filter=args.stage,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
