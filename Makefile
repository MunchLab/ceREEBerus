.PHONY: tests

html:
	# Running sphinx-build to build html files in build folder.
	rm -rf docs
	mkdir -p docs
	sphinx-build -M html doc_source docs
	rsync -a docs/html/ docs/
	rm -rf docs/html

release:
	python -m build

tests:
	# Running unittests
	@python -m unittest