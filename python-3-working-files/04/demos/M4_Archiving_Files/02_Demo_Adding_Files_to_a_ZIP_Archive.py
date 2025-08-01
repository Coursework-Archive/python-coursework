import zipfile
import os

files_to_add = ['../files/01_file.csv',
                '../files/01_file.txt']


def should_add_file(file_path):
    if not os.path.exists(file_path):
        print(f"[SKIP] {file_path} (not found)")
        return False
    if not os.path.isfile(file_path):
        print(f"[SKIP] {file_path} (not a regular file)")
        return False
    return True


def prompt_overwrite(zip_path):
    if os.path.exists(zip_path):
        ans = input(f"[PROMPT] {zip_path} already exists. Overwrite? (y/n): ")
        return ans.lower() == 'y'
    return True


def prompt_duplicate_in_archive(arcname, archive):
    if arcname in archive.namelist():
        ans = input(f"[PROMPT] {arcname} already exists in archive. Overwrite? (y/n): ")
        return ans.lower() == 'y'
    return True


def add_to_zip(zipf, files, opt, base_dir):
    if opt == 'w' and not prompt_overwrite(zipf):
        print(f"[SKIP] Zip {zipf} cancelled.")
        return

    with zipfile.ZipFile(zipf, opt, allowZip64=True) as archive:
        for f in files:
            if not should_add_file(f):
                continue

            arcname = os.path.relpath(f, start=base_dir)
            if not prompt_duplicate_in_archive(arcname, archive):
                print(f"[SKIP] Existing file in archive: {arcname}")
                continue

            try:
                archive.write(f, arcname=arcname)
                print(f"[SUCCESS] Zipped: {f} → {zipf}")
            except Exception as e:
                print(f"[SKIP] adding {f} → {zipf} - {e}")


root_dir = '../files'
zipped_file_path = '../files.zip'

add_to_zip(zipped_file_path, files_to_add, 'a', root_dir)
