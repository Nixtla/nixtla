#!/bin/bash

BASE_DIR="nbs/docs/"
SUB_DIRS=("getting-started" "tutorials" "how-to-guides")

counter=0
for sub_dir in "${SUB_DIRS[@]}"; do
    DIR="$BASE_DIR$sub_dir/"
    if [[ -d "$DIR" ]]; then
	while read -r ipynb_file; do
	    echo $counter
            md_file="${ipynb_file%.ipynb}.md"
            md_file="${md_file/docs/_docs/docs}"
            quarto render "$ipynb_file" --to md
            python -m action_files.modify_markdown --file_path "$md_file" --slug_number "$counter"
	    ((counter++))
	done < <(find "$DIR" -type f -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | sort)
    else
        echo "Directory $DIR does not exist."
    fi
done
