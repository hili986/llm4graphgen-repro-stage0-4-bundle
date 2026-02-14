PYTHON ?= python

.PHONY: check stage0_summary stage1_smoke stage1_summary stage2_rule stage3_dist stage4_property

check:
	$(PYTHON) -m pytest -q

stage0_summary:
	$(PYTHON) -m llm4graphgen.smoke --summary

stage1_smoke:
	$(PYTHON) -m llm4graphgen.stage1_smoke

stage1_summary:
	$(PYTHON) -m llm4graphgen.stage1_smoke --provider mock --run-id stage1_summary_run

stage2_rule:
	$(PYTHON) -m llm4graphgen.stage2_rule_based

stage3_dist:
	$(PYTHON) -m llm4graphgen.stage3_distribution

stage4_property:
	$(PYTHON) -m llm4graphgen.stage4_property
