# Paper templates

This repository contains two self-contained versions of the paper:

- `arr/acl_latex.tex` uses the ARR/ACL review format.
- `workshop/acl_latex.tex` uses the NeurIPS 2026 double-blind workshop format.

## Overleaf

Upload the repository (or either subfolder) to Overleaf, then choose the corresponding `acl_latex.tex` file as the **Main document** in the Overleaf project settings. Use pdfLaTeX for both versions.

Each subfolder includes its own style files, bibliography, sections, figures, tables, and appendix. The NeurIPS version also includes the required checklist.

## Local compilation

From this repository's root, run one of:

```bash
latexmk -cd -pdf arr/acl_latex.tex
latexmk -cd -pdf workshop/acl_latex.tex
```
