.PHONY: install test todo smoke experiments clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest

# Lists everything still marked TODO(verify) (the brief wants exact references in comments/docs).
todo:
	@grep -rn "TODO" src tests configs docs README.md LITERATURE.md || echo "no TODO markers"

smoke:
	psoann run configs/default.toml --runs 2 --out results/smoke

# Every shipped sweep with the default 10 runs per point (takes a few minutes).
experiments:
	for s in configs/sweeps/*.toml; do psoann sweep $$s --out results; done

clean:
	rm -rf results .pytest_cache build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
