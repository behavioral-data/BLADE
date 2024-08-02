import os
import signal
import functools
from .logger import logger


def get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_root_parent_dir():
    return os.path.abspath(os.path.join(get_root_dir(), os.pardir))


def get_logs_dir():
    return os.path.join(get_root_parent_dir(), "logs")


def get_conf_dir():
    return os.path.join(get_root_dir(), "conf")


def get_datasets_dir():
    return os.path.join(get_root_dir(), "datasets")


def get_dataset_dir(dataset_name):
    return os.path.join(get_datasets_dir(), dataset_name)


def get_dataset_csv_path(dataset_name):
    return os.path.join(get_datasets_dir(), dataset_name, "data.csv")


def get_dataset_info_path(dataset_name):
    return os.path.join(get_datasets_dir(), dataset_name, "info.json")


def get_dataset_annotations_path(dataset_name):
    return os.path.join(get_datasets_dir(), dataset_name, "annotations.csv")


def get_dataset_mcq_path(dataset_name: str):
    return os.path.join(get_datasets_dir(), dataset_name, "mcq_dataset.json")


def get_absolute_dir(directory_path):
    if not os.path.isabs(directory_path):
        return os.path.abspath(directory_path)
    return directory_path


def timeout(seconds=5, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                result = default
            finally:
                logger.debug(f"Timeout for {func.__name__}, args={args}")
                signal.alarm(0)

            return result

        return wrapper

    return decorator
