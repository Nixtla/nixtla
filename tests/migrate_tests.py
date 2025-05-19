import json
import re

pattern = r"(?ms)^#\s*\|\s*hide\s*\n.*?(?=^#\||^<VSCode\.Cell|\Z)"

# Load the Jupyter Notebook
notebook_filename = "/home/nixtla/nbs/src/nixtla_client.ipynb"
new_notebook_filename = "/home/nixtla/nbs/src/new_nixtla_client.ipynb"
with open(notebook_filename, 'r') as f:
    notebook_content = json.load(f)

# Prepare to collect test functions
test_functions = []
new_cells = []

# Iterate through the cells in the notebook
for cell in notebook_content['cells']:
    cell_text = "".join(cell['source'])
    if cell['cell_type'] == 'code':
        # Check if the cell contains a test function
        matches = re.findall(pattern, cell_text)
        if matches:
            test_functions.append(cell_text)
        else:
            new_cells.append(cell)
    else:
        # Others such as markdown cells are added directly
        test_functions.append(cell_text)
        new_cells.append(cell)

# Write the test functions to a new Python file
with open('/home/nixtla/tests/test_cases.py', 'w') as f:
    f.write('\n\n'.join(test_functions))

# Update the notebook with the new cells
notebook_content['cells'] = new_cells

# Save the modified notebook
with open(new_notebook_filename, 'w') as f:
    json.dump(notebook_content, f, indent=2)

print("Test functions have been moved to 'test_cases.py' and removed from the notebook.")
