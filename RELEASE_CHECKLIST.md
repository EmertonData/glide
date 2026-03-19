
# Release checklist 

- [ ] Use one of `make bump-major` or `make bump-minor` or `make bump-patch` to update the version number in the pyproject.toml.
- [ ] Check that `main` contains all intended changes for the release.
- [ ] Make sure CI tests pass on GitHub Actions for the latest commit on main. Otherwise fix issues in a Pull Request and merge it.
- [ ] Checkout `main` and pull latest changes: `git checkout main && git pull origin main`.
- [ ] Publish to TestPyPI to verify the build:
    * Manually trigger the TestPyPI publish job on GitHub Actions
    * Verify that the build and TestPyPI publish succeeded
    * Test installation from TestPyPI:
        - create a new empty virtual environment
        - `uv pip install --index-url https://test.pypi.org/simple/ --no-deps glide-py`
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Update `CHANGELOG.md`:
    * Move all entries from `[Next release]` into a new version section `[X.Y.Z] – YYYY-MM-DD`
    * Add a `💛 Contributors` line thanking everyone who contributed
    * Leave an empty `[Next release]` section at the top
- [ ] Tag manually the last commit with a tag of the form "vX.Y.Z"
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on tag pushes
    * Verify the package appears on PyPI
    * Test installation:
        - create a new empty virtual environment
        - `uv pip install glide-py` (you might need to run `uv cache clean` first).
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Create new release on GitHub for this tag

