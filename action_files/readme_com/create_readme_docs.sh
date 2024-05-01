#!/bin/bash

BASE_DIR="nbs/docs/"
SUB_DIRS=("1_getting_started" "2_capabilities" "3_deployment" "4_tutorials" "5_use_cases" "7_other_resources")

counter=0
for sub_dir in "${SUB_DIRS[@]}"; do
    DIR="$BASE_DIR$sub_dir/"
    if [[ -d "$DIR" ]]; then
	while read -r ipynb_file; do
	    echo $counter
            md_file="${ipynb_file%.ipynb}.md"
            md_file="${md_file/docs/_docs/docs}"
            quarto render "$ipynb_file" --to md
            python -m action_files.readme_com.modify_markdown --file_path "$md_file" --slug_number "$counter"
	    ((counter++))
	done < <(find "$DIR" -type f -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | sort)
    else
        echo "Directory $DIR does not exist."
    fi
done

# Process SDK API Reference link
echo $counter
python -m action_files.readme_com.create_sdk_reference --slug_number "$counter" --save_dir ./nbs/_docs/docs/
((counter++))

# process changelog
echo $counter
file_changelog="./nbs/_docs/docs/CHANGELOG.md"
cp ./CHANGELOG.md ${file_changelog} 
python -m action_files.readme_com.modify_markdown --file_path "$file_changelog" --slug_number "$counter"
