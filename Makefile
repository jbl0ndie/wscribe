#
# deps
#
.PHONY: deps-sync # sync and install all dependencies
deps-sync:
	uv sync --all-groups

.PHONY: deps-sync-mlx # sync and install mlx optional dependencies (Apple Silicon)
deps-sync-mlx:
	uv sync --all-groups --extra mlx

.PHONY: deps-show # list installed packages
deps-show:
	uv pip list

#
# publishing
#
.PHONY: package-publish # publish to pypi
package-publish:
	uv publish

.PHONY: package-publish-test # publish to test.pypi
package-publish-test:
	uv publish --index testpypi

.PHONY: package-build # build source and wheel
package-build:
	uv build

.PHONY: package-version-bump-patch # bump patch version (edit pyproject.toml manually or use uv version)
package-version-bump-patch:
	uv version --bump patch

.PHONY: package-version-bump-prerelease # bump prerelease version
package-version-bump-prerelease:
	uv version --bump patch --pre alpha

#
# tests
#
.PHONY: spin # run all checks
spin: typecheck lint test-quiet

.PHONY: test # run test
test:
	uv run pytest

# mypy won't report anything if you haven't type hinted/annotated your code
.PHONY: typecheck # run mypy
typecheck:
	uv run mypy .

.PHONY: lint # run linter
lint:
	#uv run ruff check .
	uv run isort . --check-only
	uv run black . --check

.PHONY: test-quiet # run test quietly
test-quiet:
	uv run pytest -q

.PHONY: test-dry-run # dry-run test, just get test names
test-dry-run:
	uv run pytest --collect-only

include Makefile.common
