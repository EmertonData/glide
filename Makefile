.PHONY: tests

venv:
	uv sync  # installs dev group only (default)
	# To also install doc dependencies: uv sync --group doc

pre-commit:
	uv run prek run --all-files

lint:
	uv run ruff check

type-check:
	uv run ty check

tests:
	uv run pytest . -vsx

coverage:
	uv run pytest -vsx \
		--cov-branch \
		--cov=. \
		--cov-report term-missing \
		--cov-report html \
		--cov-report xml \
		.

doc:
	uv sync --group doc
	uv run mkdocs build

doc-serve:
	uv sync --group doc
	uv run mkdocs serve

clean:
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -f .coverage
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +

bump-major:
uv version --bump major

bump-minor:
uv version --bump minor

bump-patch:
uv version --bump patch
