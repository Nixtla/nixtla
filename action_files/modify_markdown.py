import os
import re

import fire
from dotenv import load_dotenv

load_dotenv()


def to_snake_case(s):
    s = s.lower()
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    s = re.sub(r'\W', '_', s)
    s = re.sub(r'_+', '_', s)
    return s

def modify_markdown(
        file_path, 
        host_url=os.environ['README_HOST_URL'], 
        category=os.environ['README_CATEGORY'],
    ):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    dir_path = os.path.dirname(file_path)
    if not dir_path.endswith("/"):
        dir_path += "/"
    
    # Extract and remove the first markdown header
    pattern_header = re.compile(r'^#\s+(.*)\n+', re.MULTILINE)
    match = pattern_header.search(content)
    
    if match:
        title = match.group(1)
        content = pattern_header.sub('', content, count=1) # remove the first match
    else:
        title = 'Something Amazing'
    slug = to_snake_case(title)
    
    # Prepare the new header
    header = f"""---
title: "{title}"
slug: "{slug}"
excerpt: "Learn how to do {title} with TimeGPT"
category: {category}
hidden: false
---

    """
    
    # Remove parts delimited by ::: :::
    pattern_delimited = re.compile(r':::.*?:::', re.DOTALL)  
    content = pattern_delimited.sub('', content)

    # Modify image paths
    pattern_image = re.compile(r'!\[\]\((.*?)\)')
    content = content.replace('![figure](../../', f'![figure]({host_url}/nbs/')
    modified_content = pattern_image.sub(r'![](' + host_url + dir_path + r'\1)', content)

    # Concatenate new header and modified content
    final_content = header + modified_content
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(final_content)

if __name__=="__main__":
    fire.Fire(modify_markdown)

