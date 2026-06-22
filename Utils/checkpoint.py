import os
import datetime


def create_folder_checkpoint(path,name):
    path = path

    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)

    return path






def create_timestamp_folder(path):
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = os.path.join(path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    return path


