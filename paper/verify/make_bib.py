"""Build references.bib from registry records (doi.org content negotiation, arXiv API). Prints a verification log.
Works that cannot be fetched are NOT written. Run: python verify/make_bib.py > verify/bib_log.txt"""
import json, re, sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

DOIS = {
 "mankar2024": "10.1007/978-981-97-1488-9_7", "starner2023popsign": "10.52202/075280-0011",
 "fink2021lsfb": "10.1109/IJCNN52387.2021.9534336", "bohacek2022spoter": "10.1109/wacvw54805.2022.00024",
 "arcanjo2024": "10.5753/webmedia.2024.243245", "saggio2020": "10.3390/s20143879",
 "fawaz2019dl": "10.1007/s10618-019-00619-1", "petitjean2011dba": "10.1016/j.patcog.2010.09.013",
 "sakoe1978": "10.1109/TASSP.1978.1163055", "salvador2007fastdtw": "10.3233/IDA-2007-11508",
 "field2007cluster": "10.1111/j.1467-9868.2007.00593.x", "bragg2021fate": "10.1145/3436996",
 "bragg2019assets": "10.1145/3308561.3353774", "tran2023assets": "10.1145/3597638.3614507",
 "li2020wlasl": "10.1109/WACV45572.2020.9093512",
}
ARXIV = {"desai2023aslcitizen": "2304.05934", "vaezi2019msasl": "1812.01053", "ronchetti2023lsa64": "2310.17429",
 "jiang2021slgcn": "2103.08833", "vandendriessche2025onesh": "2502.20171", "bagnall2014nn": "1406.4757",
 "cuturi2017softdtw": "1703.01541", "desai2024biases": "2403.02563", "lugaresi2019mediapipe": "1906.08172",
 "zhang2020mphands": "2006.10214"}
NS = "{http://www.w3.org/2005/Atom}"
out, bad = [], []

def doi_bib(key, doi):
    req = urllib.request.Request("https://doi.org/" + doi, headers={"Accept": "application/x-bibtex"})
    t = urllib.request.urlopen(req, timeout=40).read().decode()
    t = re.sub(r"^@(\w+)\{[^,]+,", lambda m: "@%s{%s," % (m.group(1), key), t.strip(), count=1)
    return t

def arxiv_bib(key, aid):
    t = urllib.request.urlopen("http://export.arxiv.org/api/query?id_list=" + aid, timeout=40).read()
    e = ET.fromstring(t).find(NS + "entry")
    title = " ".join(e.find(NS + "title").text.split())
    au = " and ".join(a.find(NS + "name").text for a in e.findall(NS + "author"))
    year = e.find(NS + "published").text[:4]
    return ("@misc{%s,\n  title = {{%s}},\n  author = {%s},\n  year = {%s},\n  eprint = {%s},\n  archivePrefix = {arXiv},\n"
            "  url = {https://arxiv.org/abs/%s}\n}" % (key, title, au, year, aid, aid))

for fn, items in ((doi_bib, DOIS), (arxiv_bib, ARXIV)):
    for key, ident in items.items():
        for attempt in range(4):
            try:
                b = fn(key, ident); out.append(b)
                title = re.search(r"title\s*=\s*\{+(.+?)\}+,?\s*\n", b, re.S)
                print("OK  ", key, ident, "|", (title.group(1) if title else "?")[:90]); break
            except Exception as ex:
                time.sleep(6 * (attempt + 1))
        else:
            print("FAIL", key, ident); bad.append(key)
        time.sleep(1.5)
open("references_registry.bib", "w").write("\n\n".join(out) + "\n")
print("failed:", bad)
