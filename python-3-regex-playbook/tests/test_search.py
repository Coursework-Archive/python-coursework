import re
from pathlib import Path
from regex_playbook.search_util import compare, parts, list_matches, boundary
from regex_playbook import patterns as P

DATA = Path(__file__).parents[1] / "data" / "sample1.txt"

def test_first_match_in_file():
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            if re.search("sample", line):
                print("Found a match in line:", line.strip())
                break

def test_grouped_phone():
    text = "My phone number is 123-456-7890"
    m = re.search(P.PHONE_DASHED, text)
    assert m, "No phone match"
    print("Full:", m.group(), "Area:", m.group(1), "Exchange:", m.group(2), "Line:", m.group(3))

def test_helpers_on_literal_text():
    text = "The price is $19.99 and the date is 2023-03-11."
    compare([P.DATE_YMD, P.PRICE, P.SAMPLE_WORD], text)
    parts([r"\s"], text)
    list_matches([P.PRICE, r"\d+"], text)
    boundary(["sample"], "This is a sample line")


DATA_DIR = Path(__file__).parents[1] / "data"

TARGETS = [
    ("PHONE", re.compile(P.PHONE)),
    ("EMAIL", re.compile(P.EMAIL)),
    ("ZIP",   re.compile(P.ZIP)),
    ("ORDER", re.compile(P.ORDER)),
    ("HEX",   re.compile(P.HEX)),
]

def test_specific_patterns_all_files_in_data():
    any_found = False

    for file in sorted(DATA_DIR.iterdir()):
        if not file.is_file():
            continue
        # Try to read as text; skip binary/unicode errors gracefully
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        file_printed_header = False

        for name, rx in TARGETS:
            matches = []
            for lineno, line in enumerate(lines, 1):
                for m in rx.finditer(line):
                    matches.append((lineno, m.start(), m.end(), m.group(), line))

            if matches:
                if not file_printed_header:
                    print(f"\n📄 File: {file.name}")
                    file_printed_header = True
                print(f"  🔎 {name}: {len(matches)} match(es)")
                for lineno, start, end, text, full_line in matches:
                    print(f"    • [{lineno}:{start}-{end}] {text}")
                    print(f"      {full_line}")
                any_found = True

    # assertion: require that something matched somewhere
    assert any_found, "No matches found in data/"