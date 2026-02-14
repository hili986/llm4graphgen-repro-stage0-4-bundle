from llm4graphgen import smoke
from pathlib import Path


def test_smoke_cli_prints_ok(capsys):
    code = smoke.main([])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "ok"


def test_smoke_summary_contains_title():
    summary = smoke.build_summary(root=Path.cwd())
    assert "Stage0 结构摘要" in summary
