.PHONY: test ladder report cost
test:
	pytest
ladder:
	PYTHONPATH=src python -m trainlab.ladder --steps 30 --dataset-n 6000 --repeats 3
report:
	PYTHONPATH=src python -m trainlab.report --ledger results/ladder_cpu.json
cost:
	PYTHONPATH=src python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35
scaling:
	PYTHONPATH=src python -u -m trainlab.scaling --world-sizes 1 2 4 --steps 30 --batch 32
scaling-strong:
	PYTHONPATH=src python -u -m trainlab.scaling --world-sizes 1 2 4 --steps 30 --batch 128 --strong
ablation:
	PYTHONPATH=src python -u -m trainlab.scaling --world-sizes 1 4 --steps 30 --batch 32 --decode-cost 0 --tag ablation_nodecode
