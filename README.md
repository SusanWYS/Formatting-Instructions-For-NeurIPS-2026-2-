# ACL paper branch

This branch adapts the complete workshop paper to the official *ACL review
template from [`acl-org/acl-style-files`](https://github.com/acl-org/acl-style-files)
(source commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`).

The NeurIPS workshop source remains unchanged on the `main` branch. The ACL
version uses `paper.tex` as its main document and includes all existing paper
sections, figures, tables, references, and appendices.

## Overleaf

Upload this branch to Overleaf, choose `paper.tex` as the **Main document**, and
use pdfLaTeX.

## Local compilation

From the repository root, run:

```bash
latexmk -pdf paper.tex
```
