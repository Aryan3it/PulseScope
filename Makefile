PY ?= .venv/bin/python
USERS ?= 100000

.PHONY: setup all simulate etl analytics segment churn experiments rca rvalidate nosql report dashboard test mongo clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

all:
	$(PY) -m pulsescope.pipeline --users $(USERS)

simulate etl analytics segment churn experiments rca rvalidate nosql report:
	$(PY) -m pulsescope.pipeline $@ --users $(USERS)

mongo:
	docker compose up -d mongo

dashboard:
	.venv/bin/streamlit run dashboard/app.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf data outputs
