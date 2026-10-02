from pathlib import Path


class ModelBehaviorClassifier:
    def __init__(self, model_path: str | None) -> None:
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"behavior model not found: {model_path}")
        raise NotImplementedError("지원되는 행동 모델 형식이 아직 설정되지 않았습니다")
