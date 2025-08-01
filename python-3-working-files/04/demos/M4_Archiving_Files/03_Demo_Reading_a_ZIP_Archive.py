import zipfile
import os


def read_zip(zipf):
    if not os.path.exists(zipf):
        print(f"[SKIP] {zipf} (not found)")
        return

    with zipfile.ZipFile(zipf, 'r') as archive:
        lst = archive.namelist()
        for l in lst:
            zfinf = archive.getinfo(l)
            print(f'{l} => {zfinf.file_size} bytes, {zfinf.compress_size} compressed')


read_zip('../files.zip')
