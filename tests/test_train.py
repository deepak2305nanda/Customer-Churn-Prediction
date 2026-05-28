from src.models.train import train


def test_train_returns_reasonable_accuracy() -> None:
    result = train(random_state=42)
    assert 0.5 <= result.accuracy <= 1.0
    assert result.n_train > 0
    assert result.n_test > 0

