import os
import signal
import functools

from blade_bench.logger import logger


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


def timeout(seconds=10, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                logger.debug(f"Timeout triggered for {func.__name__}")
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError as e:
                logger.debug(f"Timeout occured: {str(e)}")
                result = default
            finally:
                # logger.debug(f"Clearing alarm for {func.__name__}")
                signal.alarm(0)

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    import time

    @timeout(seconds=10, default="Timed Out!")
    def long_running_function():
        time.sleep(12)  # Simulate long-running task
        return "Completed"

    result = long_running_function()
    r2 = long_running_function()
    r3 = long_running_function()
    r4 = long_running_function()

    print(result)
