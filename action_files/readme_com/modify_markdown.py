import os
import re
from pathlib import Path
import requests

import fire
from dotenv import load_dotenv

load_dotenv()


def to_snake_case(s):
    s = s.lower()
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    s = re.sub(r"\W", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

def modify_markdown(
    file_path,
    slug_number=0,
    host_url=os.environ["README_HOST_URL"],
    category=os.environ["README_CATEGORY"],
    api_key=os.environ["README_API_KEY"],
    readme_version=os.environ["README_VERSION"],
):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    dir_path = os.path.dirname(file_path)
    if not dir_path.endswith("/"):
        dir_path += "/"

    # Extract and remove the first markdown header
    pattern_header = re.compile(r"^#\s+(.*)\n+", re.MULTILINE)
    match = pattern_header.search(content)

    if match:
        title = match.group(1)
        content = pattern_header.sub("", content, count=1)  # remove the first match
    else:
        title = "Something Amazing"
    slug = to_snake_case(title)

    # Get category id for this doc based on the parent folder name
    url = "https://dash.readme.com/api/v1/categories"
    headers = {"authorization": f"{api_key}",
               "x-readme-version": f"{readme_version}"}    
    try:
        response = requests.get(url, headers=headers)
        categories = {category["slug"]:category["id"] for category in response.json()}
        if Path(file_path).name == 'CHANGELOG.md':  
            category_slug = 'getting-started'
            slug = category_slug + '-' + slug
        else:
            parent = Path(file_path).parents[0].name
            grandparent = Path(file_path).parents[1].name
            if grandparent == "docs":
                category_slug = parent
                slug = category_slug + '-' + slug
            else:
                category_slug = grandparent
                subcategory = parent
                slug = category_slug + '-' + subcategory + '-' + slug
        category = categories[category_slug]
    except:
        pass

    # Hide the unnecessary capabilities notebook for readme.com
    if slug == 'capabilities-capabilities':
        hidden = True
    else:
        hidden = False

    # Prepare the new header
    header = f"""---
title: "{title}"
slug: "{slug}"
order: {slug_number}
category: {category}
hidden: {hidden}
---

    """

    # Remove parts delimited by ::: :::
    pattern_delimited = re.compile(r":::.*?:::", re.DOTALL)
    content = pattern_delimited.sub("", content)

    # Modify image paths
    content = content.replace("![figure](../../", f"![figure]({host_url}/nbs/")
    pattern_image = re.compile(r"!\[\]\(((?!.*\.svg).*?)\)")
    modified_content = pattern_image.sub(
        r"![](" + host_url + dir_path + r"\1)", content
    )

    # Concatenate new header and modified content
    final_content = header + modified_content

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(final_content)


if __name__ == "__main__":
    fire.Fire(modify_markdown)
