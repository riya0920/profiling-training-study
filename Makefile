.PHONY: test ladder report cost scaling scaling-strong ablation accuracy affinity
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
accuracy:
	PYTHONPATH=src python -u -m trainlab.accuracy --target 0.85 --rate 0.35 --repeats 2 --max-steps 600 --train-n 3000 --decode-cost 80
affinity:
	PYTHONPATH=src python -u -m trainlab.affinity --repeats 7 --steps 40
