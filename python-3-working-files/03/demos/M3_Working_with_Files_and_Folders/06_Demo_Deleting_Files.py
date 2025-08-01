import os
from send2trash import send2trash


def remove_file(src: str):
    if not os.path.exists(src):
        print(f"[SKIP] Path does not exist: {src}")
        return

    if os.path.isfile(src):
        resource_type = "file"
    elif os.path.isdir(src):
        resource_type = "folder"
    else:
        print(f"[SKIP] {src} is not a file or folder. Unsupported type.")
        return

    ans = input(f"[PROMPT] Move {resource_type} '{src}' to trash? (y/n): ")
    if ans.lower() != 'y':
        print(f"[SKIP] Cancelled: {src} was not moved.")
        return

    send2trash(src)
    print(f"[SUCCESS] {resource_type.capitalize()} moved to trash: {src}")


remove_file('../files/02_test_file.txt')
