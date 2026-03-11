- [ ] Check that `main` contains all intended changes for the release.
- [ ] Make sure CI tests pass on GitHub Actions for the latest commit on master. Otherwise fix issues in a Pull Request and merge it.
- [ ] Checkout `main` and pull latest changes: `git checkout main && git pull origin main`.
- [ ] Update the version number in the pyproject.toml
- [ ] Publish to TestPyPI to verify the build:
    * Manually trigger the TestPyPI publish job on GitHub Actions
    * Verify that the build and TestPyPI publish succeeded (https://test.pypi.org/project/MAPIE/)
    * Test installation from TestPyPI:
        - create a new empty virtual environment
        - `uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match glide`
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Tag manually the last commit
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on tag pushes (might take a few minutes to start)
    * The `pypi` environment requires manual approval (configured in repo settings)
    * Approve the deployment in the GitHub Actions UI when prompted
    * Verify the package appears on PyPI after approval (https://pypi.org/project/glide/)
    * Test installation:
        - create a new empty virtual environment
        - uv pip install glide` (you might need to run `uv cache clean` first).
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Create new release on GitHub for this tag
- [ ] Check that the new stable version of the documentation is built and published and that the new version appears in the version selector (should be automatically made by a Read The Docs automation).

# Finalisation

- [ ] Add a PyPi badge on the readme linking to the pypi page
- [ ] Add a badge on the readme "commits since last release"
