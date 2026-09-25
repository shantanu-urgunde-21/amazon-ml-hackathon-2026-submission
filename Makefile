PYTHON   ?= python3
PKG      := code/business_entity_resolution
VENV     := $(PKG)/.venv
VPY      := $(CURDIR)/$(VENV)/bin/python
TEAM     ?= team
ZIP_NAME := $(TEAM)_submission.zip

.PHONY: help init run freeze package clean clean-all

help:
	@echo "make init             create $(VENV) and install requirements.txt"
	@echo "make run              run src/main.py inside the venv"
	@echo "make freeze           pin installed packages into requirements.txt"
	@echo "make package TEAM=x   build x_submission.zip"
	@echo "make clean            remove caches"
	@echo "make clean-all        remove caches and the venv"

# Reinstalls whenever requirements.txt changes
$(VENV)/.installed: $(PKG)/requirements.txt
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r $(PKG)/requirements.txt
	touch $@

init: $(VENV)/.installed

# Runs from inside the package dir, matching the README instructions
run: init
	cd $(PKG) && $(VPY) src/main.py

freeze: init
	$(VPY) -m pip freeze > $(PKG)/requirements.txt

# Submission layout: output/, code/business_entity_resolution/, Documentation_template.md
package: clean
	@test -f output/matching_results.tsv || (echo "missing output/matching_results.tsv" && exit 1)
	@test -f output/candidate_pairs.tsv  || (echo "missing output/candidate_pairs.tsv" && exit 1)
	rm -f $(ZIP_NAME)
	zip -r $(ZIP_NAME) \
		output/matching_results.tsv output/candidate_pairs.tsv \
		$(PKG) \
		Documentation_template.md \
		-x '*/.venv/*' '*/__pycache__/*' '*.pyc' '*/.DS_Store'
	@echo "built $(ZIP_NAME)"

clean:
	find $(PKG) -type d -name __pycache__ -prune -exec rm -rf {} +
	find $(PKG) -name '*.pyc' -delete

clean-all: clean
	rm -rf $(VENV)
