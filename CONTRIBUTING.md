# How to contribute

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

## Do you have a feature request?

* Ensure that it hasn't been yet implemented in the `master` branch of the repository and that there's not an Issue requesting it yet.
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

#### 1. Clone the repository
* HTTPS: `git clone https://github.com/Nixtla/nixtla.git`
* SSH: `git clone git@github.com:Nixtla/nixtla.git`
* GitHub CLI: `gh repo clone Nixtla/nixtla`

#### 2. Set up a virtual environment
The repo comes with an `requirements.txt` file which contains the libraries needed to run the api. You need to have installed virtualenv. Just use:

```
make init
```

