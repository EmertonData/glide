.PHONY: tests

venv:
	uv sync --all-groups

venv-doc:
	uv sync --group doc

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

_sync-doc:
	uv sync --group doc

doc: _sync-doc
	if [ -n "$$READTHEDOCS_OUTPUT" ]; then \
		uv run mkdocs build --site-dir "$$READTHEDOCS_OUTPUT/html"; \
	else \
		uv run mkdocs build; \
		uv run mkdocs serve; \
	fi

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
