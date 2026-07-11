
from RLfoundry.Utils.checkpoint import create_folder_checkpoint, create_timestamp_folder
import os


def run_startup(path,cfg):
    '''
        Inslsaation of the environment and folder creation
        '''
    # create the folders needed
    path = create_folder_checkpoint(path, cfg.exp_name)
    path = create_timestamp_folder(path)

    # save cfg

    txt_file_path = os.path.join(path, "config.txt")


    return path

