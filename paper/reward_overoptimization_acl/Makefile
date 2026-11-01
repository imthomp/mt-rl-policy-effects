PYTHON ?= python
LATEXMK ?= latexmk
TECTONIC ?= tectonic
.PHONY: all stats figures scan tectonic clean

all: stats figures scan
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

stats:
	cd ../.. && $(PYTHON) src/build_reward_overoptimization_paper.py

figures:
	cd ../.. && $(PYTHON) src/plot_reward_overoptimization.py
	cd ../.. && $(PYTHON) src/plot_contextual_bandit.py --output-stem paper/reward_overoptimization_acl/figures/contextual_bandit

scan:
	cd ../.. && $(PYTHON) scripts/check_tex_numbers.py paper/reward_overoptimization_acl/main.tex

tectonic: stats figures scan
	$(TECTONIC) main.tex --keep-logs --keep-intermediates

clean:
	$(LATEXMK) -C main.tex
