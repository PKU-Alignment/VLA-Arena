print-%  : ; @echo $* = $($*)
PROJECT_NAME   = vla-arena
COPYRIGHT      = "VLA-Arena Team. All Rights Reserved."
PROJECT_PATH   = vla_arena
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) scripts tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi" 2>/dev/null)
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTESTOPTS     ?=

UV ?= uv
UV_PROJECT_BASE  ?= envs/base
UV_PROJECT_LINT  ?= envs/lint
UV_PROJECT_BUILD ?= envs/build

.PHONY: default
default: install

install:
	$(UV) sync --project $(UV_PROJECT_BASE)

install-editable: install

install-e: install-editable  # alias

uninstall:
	@echo "uv-managed environment; remove .venv in the target project if needed"

build:
	$(UV) sync --project $(UV_PROJECT_BUILD)
	$(UV) run --project $(UV_PROJECT_BUILD) python -m build

# Tool setup targets (kept for backward compatibility)
pylint-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

flake8-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

py-format-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

ruff-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

mypy-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

pre-commit-install:
	$(UV) sync --project $(UV_PROJECT_LINT)
	$(UV) run --project $(UV_PROJECT_LINT) pre-commit install --install-hooks

docs-install:
	$(UV) sync --project $(UV_PROJECT_LINT)

pytest-install:
	$(UV) sync --project $(UV_PROJECT_BASE)

test-install: pytest-install

# Tests
pytest: test-install
	$(UV) run --project $(UV_PROJECT_BASE) python -c 'import $(PROJECT_PATH)' && \
	$(UV) run --project $(UV_PROJECT_BASE) python -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_PATH)" --cov-config=tests/.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) tests/

test: pytest

# Python linters
pylint: pylint-install
	$(UV) run --project $(UV_PROJECT_LINT) pylint $(PROJECT_PATH)

flake8: flake8-install
	$(UV) run --project $(UV_PROJECT_LINT) flake8 --count --show-source --statistics

py-format: py-format-install
	$(UV) run --project $(UV_PROJECT_LINT) isort --project $(PROJECT_PATH) --check $(PYTHON_FILES) && \
	$(UV) run --project $(UV_PROJECT_LINT) black --check $(PYTHON_FILES)

ruff: ruff-install
	$(UV) run --project $(UV_PROJECT_LINT) ruff check .

ruff-fix: ruff-install
	$(UV) run --project $(UV_PROJECT_LINT) ruff check . --fix --exit-non-zero-on-fix

mypy: mypy-install
	$(UV) run --project $(UV_PROJECT_LINT) mypy $(PROJECT_PATH) --install-types --non-interactive

pre-commit: pre-commit-install
	$(UV) run --project $(UV_PROJECT_LINT) pre-commit run --all-files

# Documentation
addlicense-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)
	command -v addlicense || go install github.com/google/addlicense@latest

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l mit -y 2024-$(shell date +"%Y") $(SOURCE_FOLDERS)

docstyle: docs-install
	$(UV) run --project $(UV_PROJECT_LINT) pydocstyle $(PROJECT_PATH)
	$(UV) run --project $(UV_PROJECT_LINT) doc8 docs

docs: docs-install
	$(UV) run --project $(UV_PROJECT_LINT) sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs docs/_build

clean-docs:
	rm -rf docs/_build

# Utility functions
lint: ruff flake8 py-format pylint addlicense

format: py-format-install ruff-install addlicense-install
	$(UV) run --project $(UV_PROJECT_LINT) isort --project $(PROJECT_PATH) $(PYTHON_FILES)
	$(UV) run --project $(UV_PROJECT_LINT) black $(PYTHON_FILES)
	$(UV) run --project $(UV_PROJECT_LINT) ruff check . --fix --exit-zero
	addlicense -c $(COPYRIGHT) -l mit -y 2024-$(shell date +"%Y") $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm -f tests/.coverage
	rm -f tests/coverage.xml

clean-build:
	rm -rf build/ dist/
	rm -rf *.egg-info .eggs

clean: clean-py clean-build clean-docs

# Development shortcuts
.PHONY: commit-checks
commit-checks: format lint test
