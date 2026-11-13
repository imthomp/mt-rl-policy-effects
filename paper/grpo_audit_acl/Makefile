PYTHON ?= python
LATEXMK ?= latexmk
TECTONIC ?= tectonic
.PHONY: all stats scan tectonic clean

all: stats scan
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

stats:
	cd ../.. && $(PYTHON) src/build_grpo_audit_paper.py

scan:
	cd ../.. && $(PYTHON) scripts/check_tex_numbers.py paper/grpo_audit_acl/main.tex

tectonic: stats scan
	$(TECTONIC) main.tex --keep-logs --keep-intermediates

clean:
	$(LATEXMK) -C main.tex
