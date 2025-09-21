import re
from pathlib import Path

EXTS = ['txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx']

def find_matching_files(data_dir: Path, exts=EXTS):
    pattern = re.compile(r'(?i)[\w.-]+\.(?:' + '|'.join(exts) + r')$')
    return [p.name for p in data_dir.iterdir() if p.is_file() and pattern.search(p.name)]


def test_file_names_in_data():
    data_dir = Path(__file__).parents[1] / "data"

    for ext in EXTS:
        matches = [f.name for f in data_dir.iterdir() if f.is_file() and f.name.lower().endswith("." + ext)]

        if matches:
            print(f"✅ {ext.upper()} files found in data/: {matches}")
        else:
            print(f"❌ No {ext.upper()} files found in data/")


