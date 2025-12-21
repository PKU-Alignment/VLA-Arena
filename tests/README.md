# VLA-Arena Tests

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vla_arena --cov-report=html

# Run specific test file
pytest tests/test_asset_packaging.py

# Run specific test
pytest tests/test_asset_packaging.py::TestTaskPackaging::test_pack_single_task
```

### Test markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py
├── README.md
└── test_asset_packaging.py   # Asset packaging and installation tests
```

## Writing Tests

Example test:

```python
import pytest
from vla_arena.vla_arena.utils.asset_manager import TaskPackager

def test_pack_task():
    packager = TaskPackager()
    # ... test code
```

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest --cov=vla_arena
```
