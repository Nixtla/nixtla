import os
import re

import fire
from dotenv import load_dotenv

load_dotenv()


def create_sdk_reference(
    save_dir,
    slug_number,
    host_url=os.environ["README_HOST_URL"],
    category=os.environ["README_CATEGORY"],
):
    file_path = f"{save_dir}/{slug_number}_sdk_reference.md"
    header = f"""---
title: "SDK Reference"
slug: "sdk_reference"
order: {slug_number}
type: "link"
link_url: "https://nixtla.github.io/nixtla/timegpt.html"
link_external: true
category: {category}
---

    """

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(header)


if __name__ == "__main__":
    fire.Fire(create_sdk_reference)
