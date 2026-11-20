# Shortcuts. Each target is a plain command you can also run by hand.
.PHONY: test dry-run data runs assets report all

test:          ## unit tests + synthetic smoke test (CPU, about 30 s)
	python -m pytest -q

dry-run:       ## whole pipeline on fake data with a fake model -> results_synthetic/
	bash scripts/dry_run.sh

data:          ## download/convert datasets, then audit them
	python scripts/prepare_data.py all
	python scripts/data_audit.py

runs:          ## all 36 fine-tuning runs (GPU); finished runs are skipped
	python scripts/run_experiments.py

assets:        ## integrity check, then every table, figure and report number
	python scripts/verify_results.py
	python scripts/make_report_assets.py

report:        ## compile the stand-alone preview of the Experiments and Results section
	cd report && latexmk -pdf preview.tex

all: data runs assets
