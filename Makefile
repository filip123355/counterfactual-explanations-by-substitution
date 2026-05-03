.PHONY: check format

check:
	uv run ruff check --fix src
	uv run ty check src

format:
	uv run ruff check --select I --fix src
	uv run ruff format src