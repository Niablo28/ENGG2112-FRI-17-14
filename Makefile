PY?=python

REPO:=.

.PHONY: train figures check-leakage test ui clean package

train:
	$(PY) scripts/retrain_end_to_end.py --cv 5

figures:
	$(PY) scripts/threshold_sweep.py
	$(PY) scripts/calibration_eval.py
	$(PY) scripts/fairness_slices.py

check-leakage:
	$(PY) scripts/check_leakage.py
	$(PY) scripts/check_duplicate_leakage.py

test:
	pytest -q scripts/tests

ui:
	$(PY) -m streamlit run ui/app.py

clean:
	rm -f reports/*_thresholds.csv reports/*reliability*.png reports/threshold_sweep.png

package:
	bash scripts/package_submission.sh

