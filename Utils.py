import os
from pathlib import Path


class Utils:
    @staticmethod
    def get_folder_list_in_folder(directory, count=True):
        folders = []
        for it in os.scandir(directory):
            if it.is_dir():
                folders.append(it.path)

        if count:
            return len(folders)
        else:
            return folders

    @staticmethod
    def get_files_list_in_folder(directory, count=True):
        if count:
            return len(os.listdir(directory))
        else:
            return os.listdir(directory)

    @staticmethod
    def remove_empty_folders(target_path):
        for p in Path(target_path).glob('**/*'):
            if p.is_dir() and len(list(p.iterdir())) == 0:
                os.rmdir(p)
