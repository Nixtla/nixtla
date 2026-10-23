load_docs_scripts:
# 	load processing scripts
	if [ ! -d "docs-scripts" ] ; then \
		git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch; \
	fi

api_docs:
	python docs/to_mdx.py docs

examples_docs:
	mkdir -p nbs/_extensions
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	python docs-scripts/update-quarto.py
	quarto render nbs --output-dir ../docs/mintlify/

format_docs:
	# replace _docs with docs and remove the sed that breaks LaTeX (converts \\\\ to \\)
	@if [[ "$$OSTYPE" == "darwin"* ]]; then \
		sed -i '' -e 's/_docs/docs/g' -e '/\\\\\\\\.*\\\\/d' ./docs-scripts/docs-final-formatting.bash; \
	else \
		sed -i -e 's/_docs/docs/g' -e '/\\\\\\\\.*\\\\/d' ./docs-scripts/docs-final-formatting.bash; \
	fi
	bash ./docs-scripts/docs-final-formatting.bash
	find docs/mintlify/examples -name "*.mdx" ! -name "*.html.mdx" -type f -exec sh -c 'mv "$$1" "$${1%.mdx}.html.mdx"' _ {} \;

preview_docs:
	cd docs/mintlify && mintlify dev

clean:
	find docs/mintlify -name "*.mdx" -exec rm -f {} +


all_docs: load_docs_scripts api_docs examples_docs format_docs

licenses:
	pip-licenses --format=csv --with-authors --with-urls > third_party_licenses.csv
	python scripts/filter_licenses.py
	rm -f third_party_licenses.csv
	@echo "✓ THIRD_PARTY_LICENSES.md updated"