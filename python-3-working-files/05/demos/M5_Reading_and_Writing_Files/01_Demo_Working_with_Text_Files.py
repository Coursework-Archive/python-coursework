import os


def read_txt(fn):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    with open(fn, 'r', encoding='utf-8') as f:
        print(f.read())


def read_txt_by_line(fn):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return []

    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            print(line, end='')  # avoid double newline
        return lines


def write_new_txt(fn, text):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    existing_lines = read_txt_by_line(fn)
    if any(text.strip() == line.strip() for line in existing_lines):
        ans = input(f"[PROMPT] '{text}' is already in {fn}. Add duplicate anyway? (a): ")
        if ans.lower() != 'a':
            print(f"[SKIP] Adding '{text}' cancelled")
            return

    with open(fn, 'a', encoding='utf-8') as f:
        f.write(text + '\n')
        print(f"[SUCCESS] Adding new content to {fn}.")


# Example usage:
# read_txt('../files_to_read/backup.py')
# read_txt_by_line('../files_to_read/backup.py')
# write_new_txt('../files_to_read/example.txt', 'this is a test...')
