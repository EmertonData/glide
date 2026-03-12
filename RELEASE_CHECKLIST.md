
# Release checklist 

- [ ] Check that `main` contains all intended changes for the release.
- [ ] Make sure CI tests pass on GitHub Actions for the latest commit on main. Otherwise fix issues in a Pull Request and merge it.
- [ ] Checkout `main` and pull latest changes: `git checkout main && git pull origin main`.
- [ ] Use one of `make bump-major` or `make bump-minor` or `make bump-patch` to update the version number in the pyproject.toml (see [here](https://docs.astral.sh/uv/guides/package/#updating-your-version) for details. Version must change or release will fail).
- [ ] Publish to TestPyPI to verify the build:
    * Manually trigger the TestPyPI publish job on GitHub Actions
    * Verify that the build and TestPyPI publish succeeded (https://test.pypi.org/project/glide-py/)
    * Test installation from TestPyPI:
        - create a new empty virtual environment
        - `uv pip install --index-url https://test.pypi.org/simple/ --no-deps glide-py`
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Tag manually the last commit
- [ ] Monitor the PyPI publish job on GitHub Actions:
    * The workflow automatically triggers on tag pushes (might take a few minutes to start)
    * The `pypi` environment requires manual approval (configured in environment settings as Required Reviewers)
    * Approve the deployment in the GitHub Actions UI when prompted
    * Verify the package appears on PyPI after approval (https://pypi.org/project/glide-py/)
    * Test installation:
        - create a new empty virtual environment
        - `uv pip install glide-py` (you might need to run `uv cache clean` first).
        - import glide and verify version: `python -c "import glide; print(glide.__version__)"`
- [ ] Create new release on GitHub for this tag (see [here](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository))
- [ ] Check that the new stable version of the documentation is built and published and that the new version appears in the version selector (should be automatically made by a Read The Docs automation).

