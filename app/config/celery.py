from celery import Celery, Task

from config.sentry import init_sentry
from config.settings import Settings


class CeleryConfig:
    """
    Celery configuration class.
    (cf. https://steve.dignam.xyz/2023/05/20/many-problems-with-celery/)
    """

    enable_utc = True
    timezone = "UTC"

    # Disable broker connection retry on startup
    # (cf. https://docs.celeryq.dev/en/stable/userguide/configuration.html#broker-connection-retry-on-startup)
    broker_connection_retry_on_startup = True

    # Disable prefetching
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#worker-prefetch-multiplier)
    worker_prefetch_multiplier = 1

    # Kill all long-running tasks when the connection is lost
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#worker-cancel-long-running-tasks-on-connection-loss)
    worker_cancel_long_running_tasks_on_connection_loss = True

    # Acknowledge tasks after they are done, not before
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#task-acks-late)
    task_acks_late = True

    # Reject tasks when the worker that was processing them is lost
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#task-reject-on-worker-lost)
    task_reject_on_worker_lost = True

    # Change the default queue type to quorum
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#task-default-queue-type)
    task_default_queue_type = "quorum"

    # Disable the global QoS
    # (cf. https://docs.celeryq.dev/en/latest/userguide/configuration.html#worker-detect-quorum-queues)
    worker_detect_quorum_queues = True

    task_routes = {
        "compute_analysis_task": "compute",
        "process_chat_message_task": "chat",
        "process_report_chat_message_task": "chat",
    }
    task_soft_time_limit = 25 * 60
    task_time_limit = 30 * 60


class BaseTask(Task):
    # Enable exponential backoff for retries
    #  (cf. https://docs.celeryq.dev/en/stable/userguide/tasks.html#Task.retry_backoff)
    retry_backoff = True


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or Settings()
    init_sentry(settings)

    celery_app = Celery(
        namespace="Agents",
        broker=str(settings.celery_broker_url),
        task_cls=BaseTask,
        include=["services.tasks"],
    )
    celery_app.config_from_object(CeleryConfig)
    celery_app.set_default()
    return celery_app


app = create_app()
