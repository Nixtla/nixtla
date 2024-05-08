# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/utils.ipynb 3
def colab_badge(path: str):
    from IPython.display import Markdown

    base_url = "https://colab.research.google.com/github"
    badge_svg = "https://colab.research.google.com/assets/colab-badge.svg"
    nb_url = f"{base_url}/Nixtla/nixtla/blob/main/nbs/{path}.ipynb"
    badge_md = f"[![]({badge_svg})]({nb_url})"
    display(Markdown(badge_md))

# %% ../nbs/utils.ipynb 4
import sys
from contextlib import contextmanager

# %% ../nbs/utils.ipynb 5
class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


@contextmanager
def not_run_in_colab():
    if "google.colab" in sys.modules:
        yield NoOpContext()
    else:
        yield


@contextmanager
def only_run_in_colab():
    if "google.colab" in sys.modules:
        yield
    else:
        yield NoOpContext()
