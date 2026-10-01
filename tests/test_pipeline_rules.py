import pytest

from signstudy import pipeline as P
from signstudy import run as R


def test_pick_prefers_higher_accuracy_then_earlier_option():
    cfg, _ = P._pick([({"a": 1}, 0.5), ({"a": 2}, 0.7), ({"a": 3}, 0.7)])
    assert cfg == {"a": 2}


def test_guard_refuses_without_prereg_tag(monkeypatch):
    monkeypatch.setattr(R, "git", lambda *a: "")
    with pytest.raises(SystemExit):
        R.guard_test_split()


def test_guard_refuses_dirty_tree(monkeypatch):
    def fake_git(*a):
        if a[0] == "tag":
            return "prereg-v1"
        if a[0] == "status":
            return " M signstudy/features.py"
        return ""
    monkeypatch.setattr(R, "git", fake_git)
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 0})())
    with pytest.raises(SystemExit):
        R.guard_test_split()


def test_vocabs_up_to_includes_available_size():
    assert P.vocabs_up_to(4) == [4]
    assert P.vocabs_up_to(10) == [10]
    assert P.vocabs_up_to(50) == [10, 50]


def test_step_rule_floor_and_cap():
    from signstudy.train import steps_for
    assert steps_for(10) == 200 and steps_for(10**6) == 4000 and steps_for(500, epochs=80) == 625
