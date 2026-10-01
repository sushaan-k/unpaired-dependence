"""Render the paper's unchanged TikZ figures for the project page."""

from pathlib import Path
import subprocess
import tempfile

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision/source"
OUT = ROOT / "docs/assets/figures"
FIGURES = ("overview", "atlas", "reference", "savings", "law", "pairfree")

PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb,graphicx,tikz,pgfplots}
\usepackage[scaled=0.92]{helvet}
\usepackage{sansmath}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta}
\tikzset{every picture/.append style={execute at begin picture={\sansmath}}}
\setlength{\textwidth}{160mm}
\begin{document}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for name in FIGURES:
            tex = work / f"{name}.tex"
            tex.write_text(PREAMBLE + rf"\input{{{name}_figure.tex}}" + "\n\\end{document}\n")
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={work}", str(tex)],
                cwd=SOURCE, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:])
            pdf = work / f"{name}.pdf"
            (OUT / pdf.name).write_bytes(pdf.read_bytes())
            with fitz.open(pdf) as doc:
                page = doc[0]
                page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(OUT / f"{name}.png")
            print(f"Rendered {name}")


if __name__ == "__main__":
    main()
