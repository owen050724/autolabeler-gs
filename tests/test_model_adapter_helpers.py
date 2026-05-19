import numpy as np

from autolabeler.datatypes import DetectionBox
from autolabeler.models.grounding_dino import (
    _best_class_match,
    _format_grounding_prompt_string,
    _format_grounding_prompts,
)
from autolabeler.models.sam2_segmenter import _boxes_for_sam2, _match_detection_count


def _classes():
    return [
        {"class_id": 0, "class_name": "person", "prompt": "person"},
        {"class_id": 1, "class_name": "bottle", "prompt": "a plastic bottle"},
    ]


def test_grounding_prompt_format_multiple_classes():
    prompts = _format_grounding_prompts(_classes())
    assert prompts == [["person", "a plastic bottle"]]
    assert _format_grounding_prompt_string(_classes()) == "person. a plastic bottle."


def test_best_class_match_handles_text_and_integer_labels():
    classes = _classes()
    assert _best_class_match("plastic bottle", classes)["class_id"] == 1
    assert _best_class_match(0, classes)["class_name"] == "person"


def test_sam2_boxes_are_flat_xyxy():
    det = DetectionBox(
        class_id=0,
        class_name="person",
        prompt="person",
        score=0.9,
        box_xyxy=(1, 2, 30, 40),
    )
    assert _boxes_for_sam2([det]) == [[1.0, 2.0, 30.0, 40.0]]


def test_match_detection_count_pads_empty_masks():
    masks = _match_detection_count([np.ones((5, 7), dtype=bool)], 2, (7, 5))
    assert len(masks) == 2
    assert masks[0].shape == (5, 7)
    assert masks[1].shape == (5, 7)
    assert not masks[1].any()
