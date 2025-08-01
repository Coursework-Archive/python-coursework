import zipfile
import os


def extract_file(zipf, fn, path):
    if not os.path.exists(zipf):
        print(f"[SKIP] {zipf} (not found)")
        return

    with zipfile.ZipFile(zipf, 'r') as archive:
        if fn not in archive.namelist():
            print(f"[SKIP] {fn} not found in archive")
            return

        out_path = os.path.join(path, fn)
        if os.path.exists(out_path):
            ans = input(f"[PROMPT] {out_path} already exists. Overwrite? (y/n): ")
            if ans.lower() != 'y':
                print(f"[SKIP] Extraction {fn} cancelled")
                return

        archive.extract(fn, path=path)
        print(f"[SUCCESS] Extracted: {fn} → /{path}")


def extract_all(zipf, path):
    if not os.path.exists(zipf):
        print(f"[SKIP] {zipf} (not found)")
        return

    if os.path.exists(path):
        ans = input(f"[PROMPT] {path} already exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print(f"[SKIP] Extraction {zipf} => /{path} cancelled")
            return

    with zipfile.ZipFile(zipf, 'r') as archive:
        archive.extractall(path=path)
        print(f"[SUCCESS] Extracted all → /{path}")


# extract_file('../files.zip', '01_file_test.txt', 'extracted')
extract_all('../files.zip', 'extracted')
