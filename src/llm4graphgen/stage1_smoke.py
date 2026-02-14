"""Stage1 最小闭环 CLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from llm4graphgen.parsers import parse_graph_output
from llm4graphgen.providers import MockProvider, OpenAIProvider


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage1 最小闭环：Provider + 解析器 + 日志落盘")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock", help="选择 Provider")
    parser.add_argument("--prompt", default="请输出一个 4 节点路径图，格式为 (n, [(u,v), ...])")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mock-output", default=None, help="仅 mock provider 生效，覆盖固定输出")
    parser.add_argument("--output-root", default="runs", help="运行目录根路径")
    parser.add_argument("--run-id", default=None, help="运行 ID，默认使用时间戳")
    return parser


def run_stage1(
    provider_name: str,
    prompt: str,
    model: str,
    temperature: float,
    output_root: Path,
    run_id: str,
    mock_output: str | None = None,
) -> tuple[int, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if provider_name == "mock":
        provider = MockProvider(fixed_output=mock_output)
    else:
        provider = OpenAIProvider()

    raw_output = provider.generate(prompt=prompt, model=model, temperature=temperature)
    parsed = parse_graph_output(raw_output)
    parsed_dict = parsed.to_dict()

    llm_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": provider.name,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "raw_output": raw_output,
        "parsed_result": parsed_dict if parsed.success else None,
        "parse_failure_reason": parsed.parse_failure_reason,
    }

    llm_io_path = run_dir / "llm_io.jsonl"
    with llm_io_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(llm_record, ensure_ascii=False) + "\n")

    parser_path = run_dir / "parser_result.json"
    with parser_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_dict, f, ensure_ascii=False, indent=2)

    run_log = run_dir / "run.log"
    run_lines = [
        "Stage1 运行日志",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Provider：{provider.name}",
        f"- 模型：{model}",
        f"- 温度：{temperature}",
        f"- 解析结果：{'成功' if parsed.success else '失败'}",
        f"- llm_io 落盘：{llm_io_path.as_posix()}",
        f"- 解析结果落盘：{parser_path.as_posix()}",
    ]
    if parsed.parse_failure_reason:
        run_lines.append(f"- 失败原因：{parsed.parse_failure_reason}")
    run_log.write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    return (0 if parsed.success else 1), run_dir


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id or _timestamp()
    code, run_dir = run_stage1(
        provider_name=args.provider,
        prompt=args.prompt,
        model=args.model,
        temperature=args.temperature,
        output_root=Path(args.output_root),
        run_id=run_id,
        mock_output=args.mock_output,
    )

    print(f"Stage1 运行目录：{run_dir.as_posix()}")
    if code == 0:
        print("Stage1 smoke 解析成功")
    else:
        print("Stage1 smoke 解析失败")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
