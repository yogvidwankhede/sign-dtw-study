import subprocess
import sys

import numpy as np

from signstudy import config as C
from signstudy import dataset as D
from signstudy.dataset import Video


def _videos():
    return [Video(sign=s, name=f"sg.{sn}-{s}-{i}.mp4", signer=f"sg.{sn}") for s in ("a", "b") for sn in (1, 2) for i in range(6)]


def test_reference_order_nested_and_seeded():
    v = _videos()
    o0, o0b, o1 = D.reference_order(v, ["a", "b"], 0), D.reference_order(v, ["a", "b"], 0), D.reference_order(v, ["a", "b"], 1)
    assert o0 == o0b and o0 != o1
    assert D.select_refs(o0, 2)[:1] == [o0["a"][0]]
    assert set(D.select_refs(o0, 2)) <= set(D.select_refs(o0, 5))


def test_reference_order_is_stable_across_processes():
    code = ("from signstudy import dataset as D; from signstudy.dataset import Video;"
            "v=[Video(sign=s,name=f'x{i}') for s in ('a','b') for i in range(8)];"
            "print(D.reference_order(v,['a','b'],3))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=C.ROOT).stdout for _ in range(3)}
    assert len(outs) == 1 and outs != {""}


def test_draw_queries_caps_per_signer_and_sign():
    v = _videos()
    q = D.draw_queries(v, per_signer_sign=2)
    counts = {}
    for i in q:
        counts[(v[i].signer, v[i].sign)] = counts.get((v[i].signer, v[i].sign), 0) + 1
    assert max(counts.values()) == 2 and len(counts) == 4
    assert D.draw_queries(v, 2) == q


def test_vocab_is_nested_and_deterministic():
    assert C.vocab(10) == C.sign_list()[:10] and set(C.vocab(10)) <= set(C.vocab(50)) and len(set(C.sign_list())) == 250
