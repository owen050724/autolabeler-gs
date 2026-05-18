from autolabeler.utils import parse_class_prompts


def test_comma_separated():
    out = parse_class_prompts("person, bicycle, dog")
    assert [c["class_name"] for c in out] == ["person", "bicycle", "dog"]
    assert [c["class_id"] for c in out] == [0, 1, 2]
    assert all(c["prompt"] == c["class_name"] for c in out)


def test_line_separated():
    raw = "person\nbicycle\ndog\n"
    out = parse_class_prompts(raw)
    assert [c["class_name"] for c in out] == ["person", "bicycle", "dog"]


def test_advanced_with_colon():
    raw = "bottle: a plastic bottle, a water bottle\nlaptop: a laptop computer"
    out = parse_class_prompts(raw)
    assert out[0]["class_name"] == "bottle"
    assert "a plastic bottle" in out[0]["prompt"]
    assert out[1]["class_name"] == "laptop"
    assert out[1]["prompt"] == "a laptop computer"


def test_dedup_and_empty():
    out = parse_class_prompts("dog, dog, cat,, ,dog")
    names = [c["class_name"] for c in out]
    assert names == ["dog", "cat"]
    assert [c["class_id"] for c in out] == [0, 1]


def test_empty_input():
    assert parse_class_prompts("") == []
    assert parse_class_prompts(None) == []
