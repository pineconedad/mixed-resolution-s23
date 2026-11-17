# Overleaf

1. Upload this zip as a new project (New Project > Upload Project).
2. Main document: `main.tex` (the paper). Compiler: pdfLaTeX. It compiles to 8 pages: the text ends on page 7 and the references continue on page 8.
3. Supplementary material: open the project menu, set Main document to `supp.tex`, recompile, and download that PDF separately. CVPR takes the supplementary material as its own PDF.
4. Figures are in `figures/`. Each one is drawn by a script in `scripts/` from numbers in this repository's `reports/salvation/` and `runs/salvation_a/`; no raw images are needed.
5. Before submitting: set `\def\paperID{...}` in both files to your paper ID, and keep `[review]` in `\usepackage[review]{cvpr}` for the anonymous submission.
