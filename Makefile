.PHONY: check format

check:
	uv run ruff check --fix src scripts
	uv run ty check src scripts

format:
	uv run ruff check --select I --fix src scripts
	uv run ruff format src scripts