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

##  PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Ensure that your PR includes a test that fails without your patch, and passes with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

### Local setup for working on a PR

#### Clone the repository
* HTTPS: `git clone https://github.com/Nixtla/nixtla.git`
* SSH: `git clone git@github.com:Nixtla/nixtla.git`
* GitHub CLI: `gh repo clone Nixtla/nixtla`

#### Set up an environment
Create a virtual environment to install the library's dependencies. We recommend [astral's uv](https://github.com/astral-sh/uv).
Once you've created the virtual environment you should activate it and then install the library in editable mode along with its
development dependencies.

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -Ue .[dev]

# If you plan to contribute to documentation, you will also need to install the
# distributed dependencies in addition to the dev dependencies
uv pip install -Ue .[dev,distributed]
```

#### Set Up Nixtla API Key
This library uses `python-dotenv` for development. To set up your Nixtla API key, add the following lines to your `.env` file:

```
NIXTLA_API_KEY=<your token>
```

* NOTE: You can get your Nixtla API key by logging into [Nixtla Dashboard](https://dashboard.nixtla.io/) where you can get few API calls for free. If you need more API calls for development purpose, please write to `support@nixtla.io`.

#### Install git hooks
Before doing any changes to the code, please install the git hooks that run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata (and avoid merge conflicts).
```
nbdev_install_hooks
pre-commit install
```

### Preview Changes
You can preview changes in your local browser before pushing by using the `nbdev_preview`.

### Building the library
The library is built using the notebooks contained in the `nbs` folder. If you want to make any changes to the library you have to find the relevant notebook, make your changes and then call:
```
nbdev_export
```

### Running tests
If you're working on the local interface you can just use `nbdev_test --n_workers 1 --do_print --timing`.

### Cleaning notebooks
Since the notebooks output cells can vary from run to run (even if they produce the same outputs) the notebooks are cleaned before committing them. Please make sure to run `nbdev_clean ` before committing your changes.

## Do you want to contribute to the documentation?

Docs are automatically created from the notebooks in the `nbs` folder.

### Modifying an existing doc
1. Find the relevant notebook.
2. Make your changes.
    * Do not rename the document.
    * Do not change the first header (title). The first header is used in Readme.com to create the filename. For example, a first header of `TimeGPT Subscription Plans and Pricing` in folder `getting-started` will result in the following online link to the document: `https://docs.nixtla.io/docs/getting-started-timegpt_subscription_plans_and_pricing`.
3. Run all cells.
4. Run `nbdev_preview`.
5. Clean the notebook metadata using `nbdev_clean --fname nbs/docs/[path_to_notebook.ipynb]`.
6. Add, commit and push the changes.
7. Open a PR.
8. Follow the steps under 'Publishing documentation'

### Creating a new document
1. Copy an existing jupyter notebook in a folder where you want to create a new document. This should be a subfolder of `nbs/docs`.
2. Rename the document using the following format: `[document_number]_document_title_in_lower_case.ipynb` (for example: `01_quickstart.ipynb`), incrementing the document number from the current highest number within the folder and retaining the leading zero.
3. The first header (title) is ideally the same as the notebook name (without the document number). This is because in Readme.com the first header (title) is used to create the filename. For example, a first header of `TimeGPT Subscription Plans and Pricing` of a document in folder `getting-started` will result in the following online link to the document: `https://docs.nixtla.io/docs/getting-started-timegpt_subscription_plans_and_pricing`. Thus, it is advised to keep the document name and header the same.
4. Work on your new document. Pay attention to:
    * The Google Colab link;
    * How images should be linked;
    * How the `IN_COLAB` variable is used to distinguish when the notebook is used locally vs in Google Colab.
5. Add the document to `nbs/mint.json` under the correct group with the following name `document_title_in_lower_case.html`.
6. Follow steps 3 - 8 under `Modifying an existing doc`.

### Publishing documentation
When the PR is approved, the documentation will not be visible directly. It will be visible:
1. When we make a release
2. When you manually trigger the workflows required to publish. The workflows you need to manually trigger under [Actions](https://github.com/Nixtla/nixtla/actions), in order, are:
    1. The `build-docs` workflow on branch `main`. Use the `Run workflow` button on the right and choose the `main` branch.
    2. The `Deploy to readme dot com` workflow on branch `main`. Use the `Run workflow` button on the right and choose the `main` branch.
    * After both workflows have completed (should take max. 10 minutes), check the [docs](https://docs.nixtla.io/) to see if your changes have been reflected.

It could be that on our Readme.com [docs](https://docs.nixtla.io/), the newly created document is not in the correct (sub)folder.
1. Go to the `Log In` (top right corner), log in with your Nixtla account.
2. Go to the Admin Dashboard (top right, under user-name)
3. On the left, go to `Guides`. You now see an overview of the documentation and the structure.
4. Simply drag and drop the document that is in the incorrect (sub)folder to the correct (sub)folder. The document will from hereon remain in the correct (sub)folder, even if you update its contents.

Make sure to check that our [Mintlify docs](https://nixtlaverse.nixtla.io/nixtla/docs/getting-started/introduction.html) also work as expected, and your change is reflected there too. Mintlify is commonly somewhat slower syncing the docs, so it could a bit more time to reflect the change.

### Do's and don'ts
* Don't rename documents! The filename is used statically in various files to properly index the file in the correct (sub)folder. If you rename, you're effectively creating a new document. Follow the correct procedure for creating a new document (above), and check every other document (yes, every single one) in our documentation whether there's a link now breaking to the doc you renamed.
* Check the changes / new document online in both [Readme.com](https://docs.nixtla.io/) and [Mintlify](https://nixtlaverse.nixtla.io/nixtla/docs/getting-started/introduction.html).
* Screwed up? You can hide a document in Readme.com in the Admin console, under `Guides`. Make sure to unhide it again after you've fixed your misstakes.