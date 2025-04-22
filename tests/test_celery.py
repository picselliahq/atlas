import pytest
from celery import Celery


def test_celery_configuration(celery_app: Celery) -> None:
    assert celery_app.conf.broker_url == "memory://localhost/"
    assert celery_app.conf.task_always_eager is True
    assert celery_app.conf.task_eager_propagates is True
    assert all(
        task_module in celery_app.conf.include for task_module in ["services.tasks"]
    )


def test_register_new_task(celery_app: Celery) -> None:
    @celery_app.task()
    def test_task(x: int, y: int) -> int:
        return x + y

    result = test_task.delay(x=2, y=3)
    assert result.get() == 5


def test_task_error_propagation(celery_app: Celery) -> None:
    @celery_app.task()
    def test_error_task() -> None:
        raise ValueError("Intentional error for testing")

    with pytest.raises(ValueError, match="Intentional error for testing"):
        test_error_task.delay()
