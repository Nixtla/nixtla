__all__ = ['colab_badge', 'in_colab']

import sys

def colab_badge(path: str):
    from IPython.display import Markdown, display

    base_url = "https://colab.research.google.com/github"
    badge_svg = "https://colab.research.google.com/assets/colab-badge.svg"
    nb_url = f"{base_url}/Nixtla/nixtla/blob/main/nbs/{path}.ipynb"
    badge_md = f"[![]({badge_svg})]({nb_url})"
    display(Markdown(badge_md))

def in_colab():
    return "google.colab" in sys.modules
