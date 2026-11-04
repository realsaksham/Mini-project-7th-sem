"""
Static checks for the report sources (no TeX installation needed):
missing inputs/figures, undefined refs/cites, unbalanced braces/environments,
unescaped special characters, non-ASCII characters, and a rough page estimate.

    python report/check_latex.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

problems = []


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# Expand \input recursively.
files = {}


def walk(path):
    if not path.endswith(".tex"):
        path += ".tex"
    if not os.path.exists(path):
        problems.append(f"missing \\input file: {path}")
        return
    text = load(path)
    files[path] = text
    for inc in re.findall(r"\\input\{([^}]+)\}", strip_comments(text)):
        walk(inc)


walk("thesis.tex")
all_text = "\n".join(strip_comments(t) for t in files.values())

# Figures
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text):
    if not os.path.exists(os.path.join("figures", g)):
        problems.append(f"missing figure: figures/{g}")

# Labels / refs
labels = re.findall(r"\\label\{([^}]+)\}", all_text)
dupes = {l for l in labels if labels.count(l) > 1}
for d in dupes:
    problems.append(f"duplicate label: {d}")
for r in re.findall(r"\\ref\{([^}]+)\}", all_text):
    if r not in labels:
        problems.append(f"undefined reference: {r}")
unused = [l for l in labels if f"\\ref{{{l}}}" not in all_text]

# Citations
bib = load("references.bib")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", all_text):
    cited.update(k.strip() for k in group.split(","))
for k in cited - keys:
    problems.append(f"undefined citation: {k}")
uncited = keys - cited

# Per-file syntax checks
for path, text in files.items():
    body = strip_comments(text)
    no_verb = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", body, flags=re.S)
    depth = 0
    for ch in re.sub(r"\\[{}]", "", no_verb):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            problems.append(f"{path}: unmatched closing brace")
            break
    if depth > 0:
        problems.append(f"{path}: {depth} unclosed brace(s)")
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", no_verb):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            problems.append(f"{path}: \\end{{{env}}} does not match")
    if stack and path != "thesis.tex":
        problems.append(f"{path}: unclosed environments {stack}")
    for i, line in enumerate(no_verb.splitlines(), 1):
        if re.search(r"(?<!\\)#(?!\d)", line):
            problems.append(f"{path}:{i}: unescaped #")
        if any(ord(c) > 127 for c in line):
            problems.append(f"{path}:{i}: non-ASCII character: {line.strip()[:70]}")
    # '_' outside math in plain text (rough): ignore display math, $...$, and
    # commands whose arguments are file names / keys.
    no_math = re.sub(r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
                     lambda m: "\n" * m.group(0).count("\n"), no_verb, flags=re.S)
    for i, line in enumerate(no_math.splitlines(), 1):
        stripped = re.sub(r"\$[^$]*\$", "", line)
        stripped = re.sub(r"\\(label|ref|input|includegraphics|cite|texttt|begin|end|bibliography\w*)(\[[^\]]*\])?\{[^}]*\}", "", stripped)
        stripped = re.sub(r"\\\[.*?\\\]", "", stripped)
        if re.search(r"(?<!\\)_", stripped):
            problems.append(f"{path}:{i}: '_' outside math: {line.strip()[:70]}")

# Rough page estimate (12 pt, one-and-a-half spacing typical of thesis classes)
words = len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", "", all_text)))
n_fig = len(re.findall(r"\\begin\{figure\}", all_text))
n_tab = len(re.findall(r"\\begin\{table\}", all_text))
pages = words / 330 + n_fig * 0.45 + n_tab * 0.45 + 8  # + front matter (title, prefaces, TOC/LOF/LOT)

print(f"files: {len(files)}  words: {words}  figures: {n_fig}  tables: {n_tab}  "
      f"labels: {len(labels)}  citations: {len(cited)}/{len(keys)}")
print(f"estimated length: ~{pages:.0f} pages")
if uncited:
    print("bib entries never cited:", sorted(uncited))
if unused:
    print("labels never referenced:", unused)
if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("no problems found")
