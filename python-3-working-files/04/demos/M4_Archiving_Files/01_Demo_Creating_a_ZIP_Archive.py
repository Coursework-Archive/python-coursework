import zipfile
import os

files_to_zip = ['../files/subfolder/01_file_test.csv',
                '../files/subfolder/01_file_test.txt',
                '../files/subfolder/01_test_file.csv',
                '../files/subfolder/01_test_file.txt',
                '../files/01_file_test.csv',
                '../files/01_file_test.txt']


def create_zip(zipf, files, opt, base_dir):
    if opt == 'w' and os.path.exists(zipf):
        ans = input(f"[PROMPT] {zipf} exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print(f"[SKIP] Zip {zipf} cancelled")
            return

    with zipfile.ZipFile(zipf, opt, allowZip64=True) as archive:
        for f in files:
            if not os.path.exists(f):
                print(f"[SKIP] {f} (not found)")
                continue
            if not os.path.isfile(f):
                print(f"[SKIP] {f} (not a regular file)")
                continue
            try:
                archive.write(f, arcname=os.path.relpath(f, start=base_dir))
                print(f"[SUCCESS] Zipped: {f} → {zipf}")
            except Exception as e:
                print(f"[SKIP] adding {f} -> {zipf} - {e}")


root_dir = '../files'
zipped_file_path = '../files.zip'

create_zip(zipped_file_path, files_to_zip, 'w', root_dir)
