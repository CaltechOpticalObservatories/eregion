# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
# Invoke Sphinx as a module of the active Python interpreter rather than the
# bare `sphinx-build` script, which can resolve to a stale install elsewhere
# on PATH if one exists outside the project's virtualenv.
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" api/generated

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
