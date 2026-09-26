PY_VER   := 3.12
PYTHON   ?= python$(PY_VER)
PKG      := ./code/business_entity_resolution
VENV     := $(PKG)/.venv
REQS     := $(PKG)/requirements$(if $(GPU),-gpu,).txt
VPY      := $(CURDIR)/$(VENV)/bin/python
TEAM     ?= team
ZIP_NAME := $(TEAM)_submission.zip

# Fails if the venv was built with a different Python version
CHECK_PY  = $(VPY) -c 'import sys; v="%d.%d" % sys.version_info[:2]; sys.exit(0 if v == "$(PY_VER)" else f"venv is Python {v}, expected $(PY_VER). Run: make clean-all init")'

.PHONY: help init run freeze activate package clean clean-all

help:
	@echo "make init             create $(VENV) and install requirements.txt (CPU)"
	@echo "make init GPU=1       same, with requirements-gpu.txt (FAISS on CUDA)"
	@echo "make run              run src/main.py inside the venv"
	@echo "make freeze           pin installed packages into requirements.txt"
	@echo "make package TEAM=x   build x_submission.zip"
	@echo "make clean            remove caches"
	@echo "make clean-all        remove caches and the venv"

# Reinstalls whenever requirements.txt changes
$(VENV)/.installed: $(REQS)
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(CHECK_PY)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r $(REQS)
	touch $@

init: $(VENV)/.installed
	@$(CHECK_PY)

# Runs from inside the package dir, matching the README instructions
run: init
	cd $(PKG) && $(VPY) src/main.py

freeze: init
	$(VPY) -m pip freeze > $(PKG)/requirements.txt

# make can't change the calling shell; use: eval "$(make -s activate)"
activate: init
	@echo "source $(VENV)/bin/activate"

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
