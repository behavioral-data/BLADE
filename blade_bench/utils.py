import os


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
