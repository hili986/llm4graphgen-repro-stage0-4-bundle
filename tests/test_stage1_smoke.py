import json
from pathlib import Path
import shutil

from llm4graphgen.parsers import parse_graph_output
from llm4graphgen.providers import MockProvider, OpenAIProvider
from llm4graphgen import stage1_smoke


def test_parse_graph_output_supports_tolerance_and_dedup():
    raw = "(4, [\n (0,1), (1,2), (2,3), (1,0), ])"
    result = parse_graph_output(raw)
    assert result.success is True
    assert result.n == 4
    assert result.edges == [(0, 1), (1, 2), (2, 3)]
    assert result.parse_failure_reason is None


def test_parse_graph_output_failure_reason_is_chinese():
    result = parse_graph_output("not-a-graph")
    assert result.success is False
    assert "格式错误" in (result.parse_failure_reason or "")


def test_mock_provider_returns_fixed_output():
    provider = MockProvider(fixed_output="(2, [(0,1)])")
    assert provider.generate(prompt="x", model="m", temperature=0.0) == "(2, [(0,1)])"


def test_openai_provider_requires_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        OpenAIProvider()
        raise AssertionError("未抛出缺失密钥异常")
    except ValueError as exc:
        assert "OPENAI_API_KEY" in str(exc)


def test_stage1_smoke_writes_required_files():
    output_root = Path("runs") / "test_stage1_tmp"
    if output_root.exists():
        shutil.rmtree(output_root)

    run_id = "stage1_test_run"
    exit_code, run_dir = stage1_smoke.run_stage1(
        provider_name="mock",
        prompt="请输出路径图",
        model="mock-model",
        temperature=0.0,
        output_root=output_root,
        run_id=run_id,
        mock_output="(3, [(0,1), (1,2), (1,0),])",
    )
    assert exit_code == 0
    assert run_dir == output_root / run_id

    llm_io = run_dir / "llm_io.jsonl"
    parser_result = run_dir / "parser_result.json"
    run_log = run_dir / "run.log"
    assert llm_io.exists()
    assert parser_result.exists()
    assert run_log.exists()

    with llm_io.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())
    for key in ["prompt", "model", "temperature", "raw_output", "parsed_result", "parse_failure_reason"]:
        assert key in record
    assert record["parsed_result"]["success"] is True
    assert record["parse_failure_reason"] is None

    shutil.rmtree(output_root)
