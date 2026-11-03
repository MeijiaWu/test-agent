.PHONY: install test demo swt prepare images run eval report

SPLIT ?= dev
METHOD ?= agent
WORKERS ?= 2
RUN ?= $(METHOD)_$(SPLIT)

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

# bundled toy issue; needs only an OpenAI-compatible LLM server (llama-server on :8080 by default)
demo:
	testagent demo

# official SWT-bench harness (Docker required)
swt:
	git clone https://github.com/logic-star-ai/swt-bench third_party/swt-bench
	pip install -e third_party/swt-bench

prepare:
	python bench/prepare.py

images:
	python bench/build_images.py --split $(SPLIT)

run:
	python bench/run.py --method $(METHOD) --split $(SPLIT) --workers $(WORKERS) --run $(RUN)

eval:
	python bench/evaluate.py --run $(RUN) --workers 4

report:
	python bench/report.py agent_$(SPLIT) oneshot_$(SPLIT) agent_norag_$(SPLIT) agent_noreview_$(SPLIT) agent_noexec_$(SPLIT) --ref oneshot_$(SPLIT)
