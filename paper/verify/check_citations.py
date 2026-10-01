"""Look up every cited work by DOI (Crossref) or arXiv id (arXiv API) and print what the registry says."""
import json, sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

ARXIV = {
 "li2020wlasl": "1910.11006", "vaezi2019msasl": "1812.01053", "desai2023aslcitizen": "2304.05934",
 "ronchetti2016lsa64": "2310.17429", "jiang2021slgcn": "2103.08833", "vandendriessche2025onesh": "2502.20171",
 "bagnall2014nn": "1406.4757", "fawaz2019dl": "1809.04356", "cuturi2017softdtw": "1703.01541",
 "desai2024biases": "2403.02563", "lugaresi2019mediapipe": "1906.08172", "zhang2020mphands": "2006.10214",
 "starner2023popsign": "2312.xxxx", "bohacek2022spoter": "2110.05111",
}
DOI = {
 "mankar2024": "10.1007/978-981-97-1488-9_7", "bragg2021fate": "10.1145/3436996",
 "bragg2019assets": "10.1145/3308561.3353774", "tran2023assets": "10.1145/3597638.3608395",
 "petitjean2011dba": "10.1016/j.patcog.2010.09.013", "sakoe1978": "10.1109/TASSP.1978.1163055",
 "salvador2007fastdtw": "10.3233/IDA-2007-11508", "saggio2020": "10.3390/s20143896",
 "fink2021lsfb": "10.1109/IJCNN52387.2021.9534336", "field2007cluster": "10.1111/j.1467-9868.2007.00593.x",
 "arcanjo2024": "10.5753/webmedia.2024.241666",
}
def arxiv(i):
    try:
        u = "http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(i)
        t = urllib.request.urlopen(u, timeout=30).read()
        e = ET.fromstring(t).find("{http://www.w3.org/2005/Atom}entry")
        title = " ".join(e.find("{http://www.w3.org/2005/Atom}title").text.split())
        au = [a.find("{http://www.w3.org/2005/Atom}name").text for a in e.findall("{http://www.w3.org/2005/Atom}author")]
        return f"{title} | {au[0]} et al. | {e.find('{http://www.w3.org/2005/Atom}published').text[:10]}"
    except Exception as ex:
        return f"LOOKUP FAILED ({ex})"
def crossref(d):
    try:
        m = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + d, timeout=30))["message"]
        au = m.get("author", [{}])
        return f"{m['title'][0]} | {au[0].get('family','?')} et al. | {(m.get('container-title') or ['?'])[0]} | {m.get('issued',{}).get('date-parts',[['?']])[0][0]}"
    except Exception as ex:
        return f"LOOKUP FAILED ({ex})"
for k, v in ARXIV.items():
    print(f"[arXiv {v}] {k}: {arxiv(v)}"); time.sleep(3)
for k, v in DOI.items():
    print(f"[doi {v}] {k}: {crossref(v)}"); time.sleep(0.5)
