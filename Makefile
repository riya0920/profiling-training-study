.PHONY: test ladder report cost
test:
	pytest
ladder:
	PYTHONPATH=src python -m trainlab.ladder --steps 30 --dataset-n 6000 --repeats 3
report:
	PYTHONPATH=src python -m trainlab.report --ledger results/ladder_cpu.json
cost:
	PYTHONPATH=src python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35
