.PHONY: tests

# Run pre-commit hooks, then the test suite excluding heavy integration tests.
tests:
	poetry run pre-commit run --all-files
	poetry run pytest -m "not integration"
