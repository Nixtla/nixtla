import os
import re

import fire
from dotenv import load_dotenv

load_dotenv()


def to_snake_case(s):
    s = s.lower()
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    s = re.sub(r"\W", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def merge_lines(md_text):
    code_block_pattern = re.compile(r"``` (?:python|bash)([\s\S]*?)```", re.MULTILINE)
    code_blocks = code_block_pattern.findall(md_text)
    md_text_no_code = code_block_pattern.sub("CODEBLOCK", md_text)
    lines = md_text_no_code.split("\n")
    merged_lines = []
    buffer_line = ""
    in_div_block = False
    for line in lines:
        if line.strip().lower().startswith("<div>"):
            in_div_block = True
        elif line.strip().lower().endswith("</div>"):
            in_div_block = False
        if in_div_block or line.startswith(
            ("    ", "> ", "#", "-", "*", "1.", "2.", "3.", "CODEBLOCK", "!", "[")
        ):
            if buffer_line:
                merged_lines.append(buffer_line.strip())
                buffer_line = ""
            merged_lines.append(line)
        else:
            buffer_line += line.strip() + " "
    if buffer_line:
        merged_lines.append(buffer_line.strip())
    md_text_merged = "\n".join(merged_lines)
    for code_block in code_blocks:
        md_text_merged = md_text_merged.replace(
            "CODEBLOCK", f"\n``` python\n{code_block}\n```\n", 1
        )
    return md_text_merged


def modify_markdown(
    file_path,
    slug_number=0,
    host_url=os.environ["README_HOST_URL"],
    category=os.environ["README_CATEGORY"],
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

    # Prepare the new header
    header = f"""---
title: "{title}"
slug: "{slug}"
order: {slug_number}
category: {category}
hidden: false
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
    final_content = header + merge_lines(modified_content)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(final_content)


if __name__ == "__main__":
    fire.Fire(modify_markdown)
