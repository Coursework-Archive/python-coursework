import os
import shutil
from datetime import datetime


def move_file(src: str, dst: str):
    if not os.path.exists(src):
        print(f"[SKIP] Source does not exist: {src}")
        return

    if os.path.exists(dst):
        ans = input(f"[PROMPT] {dst} exists. Rename and move? (y/n): ")
        if ans.lower() != 'y':
            print(f"[SKIP] Move cancelled: {src} → {dst}")
            return

        # Rename file with timestamp to preserve it
        if os.path.isfile(dst):
            base, ext = os.path.splitext(dst)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_name = f"{base}_backup_{timestamp}{ext}"
            os.rename(dst, new_name)
            print(f"[INFO] Renamed existing file to: {new_name}")

        elif os.path.isdir(dst):
            shutil.rmtree(dst)
            print(f"[INFO] Removed existing folder: {dst}")

    shutil.move(src, dst)
    print(f"[SUCCESS] Moved: {src} → {dst}")


move_file('../files/text.txt', '../files/test.txt')
move_file('../files/test.txt', '../files/text.txt')