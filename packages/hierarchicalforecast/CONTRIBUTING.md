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

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

## Local setup for working on a PR

### Fork and Clone the repository

Start by creating your own copy of the repository:

1. Fork the repository on GitHub to your personal account
2. Clone your fork locally using one of these methods:
   * HTTPS: `git clone https://github.com/YOUR_USERNAME/hierarchicalforecast.git`
   * SSH: `git clone git@github.com:YOUR_USERNAME/hierarchicalforecast.git`
   * GitHub CLI: `gh repo clone YOUR_USERNAME/hierarchicalforecast`

3. Add the upstream repository reference:

```sh
cd hierarchicalforecast
git remote add upstream https://github.com/Nixtla/hierarchicalforecast.git
```

4. Initialize the git submodules (required for the Eigen C++ library):

```sh
git submodule update --init --recursive
```

### Set Up a Virtual Environment with `uv`

`uv` is an [open-source package management](https://docs.astral.sh/uv/getting-started/installation/) and environment management system that runs on Windows, macOS, and Linux. Once you have `uv` installed, run:

```sh
uv venv --python 3.10
```

Then, activate your new environment:

* on MacOS / Linux:

```sh
source .venv/bin/activate
```

* on Windows:

```sh
.\.venv\Scripts\activate
```

### Create a Feature Branch

Before starting work, create a descriptive branch for your changes. Use these naming conventions:

* **Features**: `feature/descriptive-name` (e.g., `feature/new-model`)
* **Fixes**: `fix/descriptive-name` (e.g., `fix/memory-leak`)
* **Issues**: `issue/issue-number` or `issue/description` (e.g., `issue/123`)

```sh
git checkout -b feature/your-feature-name
```

### Synchronize with Upstream

Before starting work, sync your fork with the main repository to ensure you have the latest changes:

```sh
git fetch upstream
git checkout main
git merge upstream/main
```

### Install Build Prerequisites

The library includes a native C++ extension (pybind11 + Eigen) that is compiled during installation. You need a C++ compiler with C++20 and OpenMP support:

* **Linux**: `gcc`/`g++` (typically pre-installed)
* **macOS**: Xcode Command Line Tools + OpenMP runtime: `brew install libomp`
* **Windows**: MSVC with OpenMP support (via Visual Studio or Build Tools)

### Install the library

Install the library in editable mode (this compiles the C++ extension):

```sh
uv pip install -e ".[dev]"
```

### Install Pre-commit Hooks

Pre-commit hooks help maintain code quality by running checks before commits.

```bash
pre-commit install
pre-commit run --files hierarchicalforecast/*
```

## Viewing documentation locally

The new documentation pipeline relies on `mintlify` and `griffe2md`.

### Install Mintlify

> [!NOTE]
> Please install Node.js before proceeding.

```sh
npm i -g mint
```

For additional instructions, you can read about it [here](https://mintlify.com/docs/installation).

To build documentation 

```sh
make all_docs
```

Finally to view the documentation

```sh
make preview_docs
```

## Running tests

If you're working on the local interface you can just use

```sh
uv run pytest
```

## Do you want to contribute to the documentation?

* The docs are automatically generated from the docstrings in the `hierarchicalforecast` folder.
* To contribute, ensure your docstrings follow the Google style format.
* Once your docstring is correctly written, the documentation framework will scrape it and regenerate the corresponding `.mdx` files and your changes will then appear in the updated docs.
* To contribute, examples/how-to-guides, make sure you submit clean notebooks, with cleared formatted LaTeX, links and images.
* Make an appropriate entry in the `docs/mintlify/docs.json` file.
