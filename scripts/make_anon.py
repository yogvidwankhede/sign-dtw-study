"""Create an anonymous copy of the repository for double-blind review: identifying strings replaced, internal notes dropped, the
earlier self-citation blinded, and a fresh one-commit git history under a placeholder identity.
Usage: python scripts/make_anon.py [OUT_DIR]   (default ../sign-dtw-study-anon). Run, then check the reported grep hits are zero."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / (ROOT.name + "-anon")
DROP = {"HUMAN_TODO.md", "REVIEW.md", "paper/paper.pdf", "CITATION.cff", "scripts/make_anon.py", ".zenodo.json", "SUBMISSION.md"}
DROP_PREFIX = ("outreach/", "release/", "arxiv", "paper/verify/", "data/")
DROP |= {"paper/references_registry.bib"}
REPL = [
    (r"Yogvid\s+Wankhede", "Anonymous Author"), (r"Wankhede", "Anonymous"), (r"Yogvid", "Anonymous"),
    (r"yogvidwankhede", "anonymous"), (r"[\w.+-]+@[\w-]+\.[\w.]+", "anonymous@example.invalid"),
    (r"10\.5281/zenodo\.\d+", "10.0000/withheld"),
    (r"https?://orcid\.org/", ""), (r"0009-0004-2258-6775", "0000-0000-0000-0000"), (r"ORCID:\s*0000-0000-0000-0000", "ORCID: withheld"),
    (r"Washington University in St\. Louis|WashU", "[institution withheld]"),
    (r"\\citep\{mankar2024\}|\\citet\{mankar2024\}|\\cite\{mankar2024\}", r"(Anonymous, 2024)"),
]
ANON_BIB = "@misc{anonymous2024,\n  author = {Anonymous},\n  title = {Details withheld for double-blind review},\n  year = {2024}\n}\n"
PATTERNS = r"yogvid|wankhede|mankar|khawshe|thune|sahu|gmail|0009-0004|wustl|washu|ghrce|raisoni|icmsle|nagpur|10\.1007/978-981-97-1488"


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split("\n")
    for rel in filter(None, files):
        if rel in DROP or rel.startswith(DROP_PREFIX):
            continue
        src, dst = ROOT / rel, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text()
        except UnicodeDecodeError:
            shutil.copy(src, dst)
            continue
        if rel == "paper/references.bib":
            text = re.sub(r"@\w+\{mankar2024,.*?\n\}\n", ANON_BIB, text, flags=re.S)
        for pat, rep in REPL:
            text = re.sub(pat, rep, text)
        dst.write_text(text)
    # anonymous review mode for the template
    pt = OUT / "paper" / "paper.tex"
    pt.write_text(pt.read_text().replace("\\usepackage{acl}", "\\usepackage[review]{acl}"))
    (OUT / "README_ANON.md").write_text("Anonymous submission repository. Identifying information has been removed for double-blind review.\n")
    # fresh history
    env = {"GIT_AUTHOR_NAME": "Anonymous", "GIT_AUTHOR_EMAIL": "anonymous@example.invalid",
           "GIT_COMMITTER_NAME": "Anonymous", "GIT_COMMITTER_EMAIL": "anonymous@example.invalid", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=OUT, check=True)
    subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "Anonymous submission snapshot"], cwd=OUT, check=True, env=env)
    hits = subprocess.run(["grep", "-rIliE", PATTERNS, ".", "--exclude-dir=.git"], cwd=OUT, capture_output=True, text=True).stdout.split()
    print("anonymous copy:", OUT, "| files still matching identifying patterns:", len(hits))
    for h in hits[:20]:
        print("  ", h)


if __name__ == "__main__":
    main()
