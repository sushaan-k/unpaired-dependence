# Assay resolution governs the transfer of molecular dependence

This edition adds the independent blood-to-colon test to the manuscript,
supplement, figure, and derived analysis release.

- `paper.pdf`: two-column reader manuscript.
- `submission.pdf`: single-column manuscript with line numbers.
- `additional-file-1.pdf`: proofs and analysis protocols.
- `source/`: LaTeX sources and template files.
- `analysis/assay_resolution/`: retrospective reconstruction and comparisons.
- `analysis/colon_generalization/`: publicly frozen colon protocol and all results.

From `source/`, build with `latexmk -pdf reader.tex main.tex supplement.tex`.
The two analysis directories must remain siblings. Their READMEs give the
numerical verification commands and the scope of each artifact. The release
contains sufficient assay distributions and prediction tables, not raw
cell-level counts or barcodes. `SHA256SUMS` binds every included file.

The colon test retained the published blood-reference model without refitting
and passed all three prespecified comparisons in eleven physical donors.
Its 15.65% deviance reduction against individual frequencies is separate from
the earlier retrospective results. The data are from the public Mennillo et al.
colon-biopsy study, not a new experimental collection.
