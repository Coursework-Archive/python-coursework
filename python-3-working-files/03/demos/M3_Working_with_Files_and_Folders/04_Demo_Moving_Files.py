import os
import shutil


def move_files(src: str, dst: str):
    if not os.path.exists(src):
        print(f"[SKIP] Source does not exist: {src}")
        return

    if os.path.exists(dst):
        ans = input(f"[PROMPT] Destination {dst} already exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print(f"[SKIP] Move cancelled: {src} → {dst}")
            return
        # Remove destination if it's a file or folder
        if os.path.isfile(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)

    shutil.move(src, dst)
    print(f"[SUCCESS] Moved: {src} → {dst}")


# Example usage
move_files('../files/02_file_test.txt', '../files/subfolder/02_file_test.txt')
move_files('../files', '../xyz')
move_files('../xyz', '../files')
