"""Build the project page from selected, unchanged manuscript paragraphs."""

from html import escape
from pathlib import Path
import re

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexWalker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision/source"
OUT = ROOT / "docs"
URL = "https://sushaan-k.github.io/unpaired-dependence/"
REPO = "https://github.com/sushaan-k/unpaired-dependence"
TITLE = "Using unpaired measurements to reduce the number of paired cells needed to estimate dependence between modalities"
FIGURES = ("overview", "atlas", "reference", "savings", "law", "pairfree")
FIGURE_LABELS = dict(zip(("overview", "atlas", "reference", "savings", "pred", "pairfree"), range(1, 7)))
AUTHORS = (
    ("Sushaan Kandukoori", "sushaankandukoori@gmail.com"),
    ("Pranava Kumar", "pranavak@mit.edu"),
    ("Shrikrishna Ramesh", "shrikrish.ramesh@gmail.com"),
)

main = (SOURCE / "main.tex").read_text()
macros = {}
for path in SOURCE.glob("*macros.tex"):
    macros.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", path.read_text()))
refs = dict(re.findall(r"siref@([^\\]+)\\endcsname\{([^}]+)\}", (SOURCE / "si_refs.tex").read_text()))
converter = LatexNodes2Text(math_mode="verbatim")


def group_after(text, command):
    start = text.index("{", text.index(command))
    node, _, _ = LatexWalker(text).get_latex_braced_group(start)
    return node.latex_verbatim()[1:-1]


def plain(text):
    text = re.sub(r"\\(?:cite|citenum)\{[^}]+\}", "", text)
    text = re.sub(r"\\Sref\{([^}]+)\}", lambda m: refs[m[1]], text)
    text = re.sub(r"\\ref\{fig:([^}]+)\}", lambda m: str(FIGURE_LABELS[m[1]]), text)
    for _ in range(3):
        text = re.sub(r"\\([A-Za-z]+)(?:\{\})?", lambda m: macros.get(m[1], m[0]), text)
    return " ".join(converter.latex_to_text(text).split())


def paragraph(text):
    return f"<p>{escape(plain(text))}</p>"


def resources():
    return f"""<div class="publication-links" aria-label="Paper resources">
      <a href="assets/papers/reading-copy.pdf">Paper</a>
      <a href="assets/papers/supplement.pdf">Supplement</a>
      <a href="{REPO}">Code</a>
      <a href="{REPO}/blob/main/extension/DATA.md">Data</a>
    </div>"""


abstract = plain(group_after(main, r"\abstract"))
sections = re.split(r"\\subsection\{([^}]+)\}", main.split(r"\section{Results}", 1)[1].split(r"\section{Discussion}", 1)[0])
selected = ([0, 1], [0, 1, 2, 4], [0, 1], [0, 1], [0, 1, 2, 3], [0, 2])
content = []
for index, (name, chosen) in enumerate(zip(FIGURES, selected)):
    heading, body = sections[1 + 2 * index:3 + 2 * index]
    prose, figure = body.split(r"\begin{figure*}", 1)
    paragraphs = [p.strip() for p in prose.strip().split("\n\n") if p.strip()]
    caption = plain(group_after(figure, r"\caption"))
    content.append(f"""<section class="section" id="{name}">
      <div class="reading"><h2>{escape(heading)}</h2>
        {''.join(paragraph(paragraphs[i]) for i in chosen)}
      </div>
      <figure>
        <a class="figure-link" href="assets/figures/{name}.png" data-figure aria-label="Enlarge Figure {index + 1}">
          <img src="assets/figures/{name}.png" alt="{escape(caption.split('. ')[0], quote=True)}" loading="lazy">
        </a>
        <figcaption><span class="figure-number">Figure {index + 1}.</span> {escape(caption)}
          <a class="vector-link" href="assets/figures/{name}.pdf">PDF</a>
        </figcaption>
      </figure>
    </section>""")

discussion = main.split(r"\section{Discussion}", 1)[1].split(r"\subsection*{Limitations}", 1)[0].strip()
limitations = main.split(r"\subsection*{Limitations}", 1)[1].split(r"\subsection*{Conclusions}", 1)[0].strip()
conclusion = main.split(r"\subsection*{Conclusions}", 1)[1].split(r"\section{Methods}", 1)[0].strip()
names = ", ".join(f'<span>{name}<sup>{i}</sup></span>' for i, (name, _) in enumerate(AUTHORS, 1))
emails = "".join(f'<a href="mailto:{email}"><sup>{i}</sup>{email}</a>' for i, (_, email) in enumerate(AUTHORS, 1))
head = f"""<meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{TITLE}</title>
  <meta name="description" content="{escape(abstract, quote=True)}">
  <meta name="theme-color" content="#ffffff">
  <link rel="canonical" href="{URL}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="{TITLE}">
  <meta property="og:description" content="{escape(abstract, quote=True)}">
  <meta property="og:url" content="{URL}">
  <meta property="og:image" content="{URL}assets/social-preview.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="{TITLE}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{TITLE}">
  <meta name="twitter:image" content="{URL}assets/social-preview.png">
  <link rel="icon" href="data:,">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css?family=Google+Sans:400,500,700|Noto+Sans:400,600" rel="stylesheet">
  <link rel="stylesheet" href="assets/style.css">
  <script>window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}}}};</script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>"""

OUT.mkdir(exist_ok=True)
(OUT / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head>{head}</head><body>
  <a class="skip-link" href="#abstract">Skip to content</a>
  <header class="publication-hero">
    <h1>Using <span>unpaired measurements</span> to reduce the number of paired cells needed to estimate dependence between modalities</h1>
    <p class="authors">{names}</p>
    <div class="author-emails">{emails}</div>
    {resources()}
  </header>
  <nav class="section-nav" aria-label="Contents">
    <a href="#abstract">Abstract</a><a href="#overview">Estimator</a><a href="#atlas">Validation</a>
    <a href="#savings">Benchmark</a><a href="#law">Theory</a><a href="#discussion">Discussion</a>
  </nav>
  <main>
    <section id="abstract" class="section abstract"><div class="reading"><h2>Abstract</h2><p>{escape(abstract)}</p></div></section>
    {''.join(content)}
    <section id="discussion" class="section"><div class="reading"><h2>Discussion</h2>
      {''.join(paragraph(p) for p in discussion.split(chr(10) * 2))}
      <details class="limitations"><summary>Limitations</summary>
        {''.join(paragraph(p) for p in limitations.split(chr(10) * 2))}
      </details>
      <h2 class="conclusions">Conclusions</h2>{paragraph(conclusion)}
    </div></section>
  </main>
  <footer>{resources()}<a href="{REPO}/tree/main/revision/source">Manuscript source</a></footer>
  <dialog class="figure-dialog" aria-label="Enlarged figure">
    <button type="button" class="close-figure" aria-label="Close figure">&times;</button>
    <img alt="">
  </dialog>
  <script src="assets/site.js"></script>
</body></html>
""")
(OUT / ".nojekyll").touch()
(OUT / "assets/social-preview.html").write_text(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><link rel="icon" href="data:,"><link href="https://fonts.googleapis.com/css?family=Google+Sans:400,500" rel="stylesheet">
<style>*{{box-sizing:border-box}}body{{margin:0;width:1200px;height:630px;display:flex;flex-direction:column;justify-content:center;padding:60px 75px;color:#303236;background:white;font-family:'Google Sans',Arial,sans-serif;letter-spacing:0}}h1{{font-size:54px;line-height:1.2;font-weight:500;margin:0 0 44px}}span{{color:#1769aa}}p{{font-size:24px;line-height:1.6;margin:0}}</style>
</head><body><h1>Using <span>unpaired measurements</span> to reduce the number of paired cells needed to estimate dependence between modalities</h1><p>{' &middot; '.join(name for name, _ in AUTHORS)}</p></body></html>""")
print("Built project page from manuscript text and numerical macros")
