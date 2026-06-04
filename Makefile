.PHONY: tests

UNIT_DIRS = glide tests/unit
FUNCTIONAL_DIRS = tests/functional

venv:
	uv sync --all-groups

venv-doc:
	uv sync --group doc

pre-commit:
	uv run prek run --all-files

lint:
	uv run ruff check --fix

type-check:
	uv run ty check
	
unit-tests:
	uv run pytest $(UNIT_DIRS) -vsx

functional-tests:
	uv run pytest $(FUNCTIONAL_DIRS) -vsx

tests: unit-tests functional-tests

coverage:
	uv run pytest -vsx \
		--cov-branch \
		--cov=glide \
		--cov-report term-missing \
		--cov-report html \
		--cov-report xml \
		$(UNIT_DIRS)

_sync-doc:
	uv sync --group doc

doc: _sync-doc
	if [ -n "$$READTHEDOCS_OUTPUT" ]; then \
		uv run mkdocs build --site-dir "$$READTHEDOCS_OUTPUT/html"; \
	else \
		uv run mkdocs serve; \
	fi

branch:
	@test -n "$(name)" || (echo "Usage: make branch name=<branch-name>"; exit 1)
	git checkout main && git pull && git checkout -b $(name)
  
test-notebooks:
	@find docs -name "*.ipynb" ! -path "*/generated/*" | while read notebook; do \
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
	rm -rf docs/generated_examples
	find . -type d -name "__pycache__" -exec rm -rf {} +

build:
	rm -rf dist
	uv build

bump-major:
	uv run bump-my-version bump major

bump-minor:
	uv run bump-my-version bump minor

bump-patch:
	uv run bump-my-version bump patch
