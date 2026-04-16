.PHONY: tests

venv:
	uv sync --all-groups

venv-doc:
	uv sync --group doc

pre-commit:
	uv run prek run --all-files
	make test-notebooks

lint:
	uv run ruff check --fix

format:
	uv run ruff format

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
		uv run mkdocs serve; \
	fi

test-notebooks:
	@find docs -name "*.ipynb" | while read notebook; do \
		echo "Testing $$notebook..."; \
		if ! uv run papermill "$$notebook" /dev/null > /tmp/nb_$$.log 2>&1; then \
			cat /tmp/nb_$$.log | grep -v -E "(UserWarning|Executing:|Output Notebook|warnings.warn)"; \
			exit 1; \
		fi; \
	done && printf "test-notebooks%-50s\033[32mPassed\033[0m\n" | tr ' ' '.'

clean:
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -f .coverage
	rm -rf htmlcov
	rm -rf site
	find . -type d -name "__pycache__" -exec rm -rf {} +

build:
	rm -rf dist
	uv build

bump-major:
	uv version --bump major

bump-minor:
	uv version --bump minor

bump-patch:
	uv version --bump patch
