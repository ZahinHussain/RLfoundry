
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

    with open(txt_file_path, "w") as f:
        f.write(f"# Config saved for experiment: {cfg.exp_name}\n")
        f.write("=" * 40 + "\n")
        for key, value in cfg.items():
            line = f"{key}: {value}"
            print(line)
            f.write(line + "\n")

    # variables needed

    return path

