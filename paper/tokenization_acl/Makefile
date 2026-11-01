PYTHON ?= python
LATEXMK ?= latexmk
TECTONIC ?= tectonic

.PHONY: all stats scan tectonic clean

all: stats scan
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

tectonic: stats scan
	$(TECTONIC) main.tex --keep-logs --keep-intermediates

stats:
	cd ../.. && $(PYTHON) src/build_tokenization_paper_stats.py

scan:
	cd ../.. && $(PYTHON) scripts/check_tex_numbers.py paper/tokenization_acl/main.tex

clean:
	$(LATEXMK) -C main.tex
