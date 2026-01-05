# How to contribute

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

## Do you have a feature request?

* Ensure that it hasn't been yet implemented in the `main` branch of the repository and that there's not an Issue requesting it yet.
* Open a new issue and make sure to describe it clearly, mention how it improves the project and why its useful.

## Do you want to fix a bug or implement a feature?

Bug fixes and features are added through pull requests (PRs).

## Prerequisites

Before contributing, ensure you have:

* A GitHub account
* Basic understanding of git and GitHub
* `uv` package manager installed (see [installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Git Fork-and-Pull Workflow

We use a fork-and-pull workflow for contributions. Here's how to get started:

### 1. Fork and clone the repository

1. Fork the repository on GitHub to your account
2. Clone your fork locally:
   * HTTPS: `git clone https://github.com/YOUR_USERNAME/mlforecast.git`
   * SSH: `git clone git@github.com:YOUR_USERNAME/mlforecast.git`
   * GitHub CLI: `gh repo clone YOUR_USERNAME/mlforecast`
3. Add the upstream repository:
   ```sh
   cd mlforecast
   git remote add upstream https://github.com/Nixtla/mlforecast.git
   ```

### 2. Create a branch

Create a branch for your changes using one of these naming conventions:

* **Feature branches**: `feature/descriptive-name` (e.g., `feature/new-model`)
* **Fix branches**: `fix/descriptive-name` (e.g., `fix/memory-leak`)
* **Issue branches**: `issue/issue-number` or `issue/description` (e.g., `issue/123`)

```sh
git checkout -b feature/your-feature-name
```

### 3. Keep your fork synchronized

Before starting work, sync with upstream:

```sh
git fetch upstream
git checkout main
git merge upstream/main
```

## Local setup for development

### 1. Set up your environment with uv

Create and activate a virtual environment:

```sh
# Create virtual environment (optionally specify Python version)
uv venv --python 3.10

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.\.venv\Scripts\activate
```

### 2. Install the library in editable mode

Install mlforecast with development dependencies:

```sh
uv pip install -e ".[dev]"
```

### 3. Optional dependencies

You can install additional optional dependencies:

```sh
uv pip install -e ".[dask,ray,spark,aws,gcp,azure,polars]"
```

### 4. Set up pre-commit hooks

Install pre-commit hooks to ensure code quality:

```sh
pre-commit install
```

## Running tests

### Standard test suite

To run the test suite (excluding Ray tests due to uv incompatibility):

```sh
uv run pytest --ignore=tests/distributed_ray
```

### Running all tests (including Ray)

If you need to run the Ray tests, you'll need to source the virtual environment directly instead of using `uv run`:

```sh
# First, ensure you're in the activated virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.\.venv\Scripts\activate  # On Windows

# Then run pytest directly
pytest
```

Note: Ray tests are excluded from `uv run pytest` due to incompatibility between uv and Ray's testing framework.

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Ensure that your PR includes a test that fails without your patch, and passes with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

## Contributing to documentation

The documentation pipeline uses `quarto`, `mintlify` and `griffe2md`.

### Install quarto

Install `quarto` from [this link](https://quarto.org/docs/get-started/)

### Install mintlify

> [!NOTE]
> Please install Node.js before proceeding.

```sh
npm i -g mint
```

For additional instructions, read [this link](https://mintlify.com/docs/installation).

### Build and preview documentation

```sh
# Install development dependencies if not already installed
uv pip install -e ".[dev]"

# Generate all documentation
make all_docs

# Preview documentation locally
make preview_docs
```

### Documentation guidelines

* The docs are automatically generated from the docstrings in the `mlforecast` folder.
* Ensure your docstrings follow the Google style format.
* Once your docstring is correctly written, the documentation framework will scrape it and regenerate the corresponding `.mdx` files and your changes will then appear in the updated docs.
* To contribute examples/how-to-guides, make sure you submit clean notebooks, with cleared formatted LaTeX, links and images.
* Make an appropriate entry in the `docs/mintlify/mint.json` file.

## Need help?

Feel free to reach out on our community Slack or open a discussion on GitHub if you have questions!
