import json

from harm_route.lang_data import final_prompts

LANGS = {"eng_Latn": {"source": "original"}, "srp_Cyrl": {"source": "mt"},
         "srp_Latn": {"source": "transform", "from": "srp_Cyrl", "transform": "srp_cyrl_to_latn"}}


def test_final_prompts(tmp_path_factory):
    root = tmp_path_factory.mktemp("e01")
    tmp_path = root / "mt" / "sys"
    tmp_path.mkdir(parents=True)
    (tmp_path / "eng_Latn.json").write_text(json.dumps({"meta": {}, "a": ["x0", "x1", "x2"]}))
    (tmp_path / "srp_Cyrl.json").write_text(json.dumps({"meta": {}, "a": ["жа", "ша", "ча"]}))
    qc = {"a": [{"chrf": 80.0}, {"chrf": 90.0}, {"chrf": 10.0}]}  # idx 2 flagged
    (tmp_path / "srp_Cyrl_qc.json").write_text(json.dumps(qc))
    rev = root / "review" / "sys"
    rev.mkdir(parents=True)
    (rev / "srp_Cyrl__r1.jsonl").write_text(
        json.dumps({"key": "a|0", "decision": "fix", "text": "џа"}) + "\n")
    p, keep, why = final_prompts(root, "sys", list(LANGS), LANGS, ["a"], chrf_flag=40)
    assert keep == {"a": [0, 1]}  # unreviewed + chrF < 40 -> dropped in all languages
    assert p["srp_Cyrl"]["a"][0] == "џа"  # human fix applied
    assert p["srp_Latn"]["a"][0] == "dža"  # transform language inherits the fix
    assert why == {"eng_Latn": 0, "srp_Cyrl": 1, "srp_Latn": 0}

    # a later review overrides the chrF rule
    with (rev / "srp_Cyrl__r1.jsonl").open("a") as f:
        f.write(json.dumps({"key": "a|2", "decision": "ok"}) + "\n")
    assert final_prompts(root, "sys", list(LANGS), LANGS, ["a"], 40)[1] == {"a": [0, 1, 2]}
