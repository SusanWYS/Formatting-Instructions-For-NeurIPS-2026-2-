# Paper formats

This repository contains the paper in two formats:

- `paper.tex` uses the NeurIPS 2026 double-blind workshop format.
- `acl/acl_latex.tex` uses the official ACL review format.

## Overleaf

To switch formats in the same Overleaf project, open **Menu**, change **Main
document**, and select either `paper.tex` or `acl/acl_latex.tex`. Use pdfLaTeX
for both versions. The two entry points share the bibliography, figures, and
generated results so these assets stay synchronized.

## Local compilation

From the repository root, run either:

```bash
latexmk -pdf paper.tex
latexmk -pdf acl/acl_latex.tex
```
