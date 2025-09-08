import re
from pathlib import Path

comment_pat = re.compile(r"<!--.*?-->", re.DOTALL)
anchor_pat = re.compile(r"<a.*?>(.*?)</a>")
output_path = Path("docs/mintlify")

# process docs
for file in Path("docs").glob("*.md"):
    text = file.read_text()
    text = comment_pat.sub("", text)
    text = anchor_pat.sub("", text)
    module_name = file.name.split(".")[-2]
    output_file = output_path / (module_name + ".mdx")
    output_file.write_text(text)

header = """---
description: Probabilistic hierarchical forecasting with statistical and econometric methods
title: "Hierarchical Forecast 👑"
---
"""
readme_text = Path("README.md").read_text()
readme_text = re.sub(r"https?://nixtlaverse\.nixtla\.io/", "./", readme_text)
readme_text = header + readme_text
(output_path / "index.mdx").write_text(readme_text)
