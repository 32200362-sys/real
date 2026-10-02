from types import SimpleNamespace

from smart_coaster.vision.yolo_cup_detector import YoloCupDetector


class FakeModel:
    names = {2: "bottle", 9: "cup"}

    def predict(self, **kwargs):
        boxes = SimpleNamespace(
            cls=[2, 9], conf=[0.99, 0.8],
            xyxy=[[0, 0, 4, 4], [1, 2, 11, 22]],
        )
        return [SimpleNamespace(boxes=boxes)]


def test_detector_finds_dynamic_cup_id_and_filters_other_classes():
    result = YoloCupDetector("unused", backend=FakeModel()).detect(object())
    assert len(result) == 1 and result[0].class_id == 9


def test_detector_rejects_model_without_cup():
    backend = FakeModel()
    backend.names = {0: "person"}
    try:
        YoloCupDetector("unused", backend=backend)
    except ValueError as exc:
        assert "cup" in str(exc)
    else:
        raise AssertionError("ValueError not raised")
