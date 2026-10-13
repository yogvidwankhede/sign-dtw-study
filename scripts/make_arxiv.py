"""Build a self-contained arXiv source folder from paper/: flattened paths, precompiled .bbl, template files, abstract text.
Usage: python scripts/make_arxiv.py [OUT_DIR]   (default: arxiv/ at the repo root, git-ignored). Needs tectonic."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "arxiv"


def expand_macros(text, numbers):
    for name, val in sorted(numbers.items(), key=lambda kv: -len(kv[0])):
        text = text.replace("\\" + name + "{}", val).replace("\\" + name, val)
    return text


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "sections").mkdir(parents=True)
    (OUT / "tables").mkdir()
    (OUT / "figures").mkdir()
    P = ROOT / "paper"
    for f in ("acl.sty", "acl_natbib.bst", "numbers.tex", "numbers_aslc.tex", "references.bib"):
        shutil.copy(P / f, OUT / f)
    for sec in (P / "sections").glob("*.tex"):
        t = sec.read_text()
        t = re.sub(r"\.\./results_aslc/tables/", "tables/aslc_", t)
        t = re.sub(r"\.\./results/tables/", "tables/", t)
        t = re.sub(r"\.\./figures_aslc/", "figures/aslc_", t)
        t = re.sub(r"\.\./figures/", "figures/", t)
        (OUT / "sections" / sec.name).write_text(t)
    for d, pre in ((ROOT / "results" / "tables", ""), (ROOT / "results_aslc" / "tables", "aslc_")):
        for f in d.glob("*.tex"):
            shutil.copy(f, OUT / "tables" / (pre + f.name))
    for d, pre in ((ROOT / "figures", ""), (ROOT / "figures_aslc", "aslc_")):
        for f in d.glob("accuracy_vs_k.pdf"):
            shutil.copy(f, OUT / "figures" / (pre + f.name))
    (OUT / "paper.tex").write_text((P / "paper.tex").read_text())
    subprocess.run(["tectonic", "-X", "compile", "paper.tex", "--keep-intermediates", "--keep-logs"], cwd=OUT, check=True,
                   capture_output=True)
    # arXiv wants the .bbl instead of the .bib
    (OUT / "references.bib").unlink()
    nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (P / "numbers.tex").read_text()))
    nums.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (P / "numbers_aslc.tex").read_text()))
    abstract = (OUT / "sections" / "abstract.tex").read_text()
    abstract = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", abstract)
    abstract = expand_macros(abstract, nums)
    abstract = abstract.replace("\\%", "%").replace("$k$", "k").replace("\\texttt", "").replace("{}", "").replace("--", "-")
    abstract = re.sub(r"\s+", " ", abstract).strip()
    (OUT / "METADATA.txt").write_text(
        "TITLE: When Does Training-Free DTW Match Learned Models on Hand Landmarks? A Data-Efficiency and Signer-Independence Study of Isolated Sign Recognition\n"
        "AUTHORS: Yogvid Wankhede (Independent Researcher; ORCID 0009-0004-2258-6775)\n"
        "CATEGORIES: cs.CV (primary) or cs.LG; cross-list cs.CL, cs.HC (author to choose)\n"
        "COMMENTS: 12 pages, 5 tables, 2 figures. Code and aggregate results: https://github.com/yogvidwankhede/sign-dtw-study, https://doi.org/10.5281/zenodo.23351799\n"
        "LICENSE: author to choose (CC BY 4.0 is common)\n\nABSTRACT:\n" + abstract + "\n")
    # intermediates not needed for upload
    for f in OUT.glob("paper.*"):
        if f.suffix in (".aux", ".log", ".blg", ".out", ".pdf"):
            f.unlink()
    print("arXiv folder:", OUT)


if __name__ == "__main__":
    main()
