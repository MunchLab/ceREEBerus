lint:
	@flake8 cereeberus/

format:
	@black cereeberus/

.PHONY: docs
docs:
	# Running sphinx-build to build html files in build folder.
	rm -rf docs
	mkdir -p docs
	sphinx-build -M html doc_source docs
	rsync -a docs/html/ docs/
	rm -rf docs/html

install-editable:
	@pip install -e .

release:
	python -m build

.PHONY: tests
tests:
	# Running unittests
	@python -m unittest

install-requirements:
	@pip install -r requirements.txt

all: install-requirements install-editable format tests docs 

help: 
	@echo "Makefile commands:"
	@echo "  make lint                - Run flake8 linter on the code"
	@echo "  make format              - Format code using black"
	@echo "  make install-editable     - Install the package in editable mode for development"
	@echo "  make docs                - Build the documentation in the docs/ folder"
	@echo "  make release             - Build the package for release"
	@echo "  make tests               - Run the unit tests"
	@echo "  make install-requirements- Install all dependencies from requirements.txt"
	@echo "  make all       - Run format, tests, docs, and release"
	@echo "  make help      - Show this help message"