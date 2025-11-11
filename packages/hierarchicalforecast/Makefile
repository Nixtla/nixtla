load_docs_scripts:
# 	load processing scripts
	if [ ! -d "docs-scripts" ] ; then \
		git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch; \
	fi

api_docs:
	lazydocs .hierarchicalforecast --no-watermark
	python docs/to_mdx.py

examples_docs:
	mkdir -p nbs/_extensions
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	python docs-scripts/update-quarto.py
	quarto render nbs --output-dir ../docs/mintlify/

format_docs:
	# replace _docs with docs
	sed -i -e 's/_docs/docs/g' ./docs-scripts/docs-final-formatting.bash
	bash ./docs-scripts/docs-final-formatting.bash


preview_docs:
	cd docs/mintlify && mintlify dev

clean:
	rm -f docs/*.md
	find docs/mintlify -name "*.mdx" -exec rm -f {} +


all_docs: load_docs_scripts api_docs examples_docs format_docs

licenses:
	pip-licenses --format=markdown --with-authors --with-urls | grep -E "GPL|AGPL|LGPL|MPL" > THIRD_PARTY_LICENSES.md