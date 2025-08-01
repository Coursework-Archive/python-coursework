import os
import shutil

def copy_file(src, dst):
    if os.path.exists(dst):
        ans = input(f"[FILE] {dst} already exists. Overwrite this file? (y/n): ")
        if ans.lower() == 'y':
            shutil.copy(src, dst)
            print(f"File overwritten: {dst}")
        else:
            print("Skipped copying file.")
    else:
        shutil.copy(src, dst)
        print(f"Copied file to {dst}")


def copy_folder(src, dst):
    if os.path.exists(dst):
        ans = input(f"[FOLDER] {dst} already exists. Overwrite this folder and all its contents? (y/n): ")
        if ans.lower() == 'y':
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Folder overwritten: {dst}")
        else:
            print("Skipped copying folder.")
    else:
        shutil.copytree(src, dst)
        print(f"Copied folder to {dst}")


# Example usage
copy_file('../files/02_file.txt', '../files/subfolder/02_file.txt')
copy_folder('../files', '../files/new_folder')
