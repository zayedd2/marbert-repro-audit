# Shortcuts. Each target is a plain command you can also run by hand.
.PHONY: test dry-run verify assets report

test:          ## unit tests, regression test on the real runs, synthetic end-to-end test (CPU, ~20 s)
	python -m pytest -q

verify:        ## re-score every saved prediction set and write results/MANIFEST.sha256
	python scripts/verify_results.py

assets:        ## every table, figure and report number, plus the cross-check against the team's files
	python scripts/make_report_assets.py

dry-run:       ## whole analysis pipeline on fake runs -> results_synthetic/
	bash scripts/dry_run.sh

report:        ## compile the stand-alone preview of the Experiments and Results layout
	cd report && latexmk -pdf preview.tex
