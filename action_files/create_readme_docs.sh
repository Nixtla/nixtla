#!/bin/bash

BASE_DIR="nbs/docs/"
SUB_DIRS=("tutorials" "getting-started" "how-to-guides")

for sub_dir in "${SUB_DIRS[@]}"; do
    DIR="$BASE_DIR$sub_dir/"
    if [[ -d "$DIR" ]]; then
 	find "$DIR" -type f -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | while read -r ipynb_file; do
            md_file="${ipynb_file%.ipynb}.md"
            md_file="${md_file/docs/_docs/docs}"
            quarto render "$ipynb_file" --to md
            python -m action_files.modify_markdown --file_path "$md_file"
        done
    else
        echo "Directory $DIR does not exist."
    fi
done
