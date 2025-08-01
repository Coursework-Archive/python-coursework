import json
import os


def read_print_json(fn, pretty, sort):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    with open(fn) as json_file:
        data = json.load(json_file)
        print(json.dumps(data, sort_keys=sort, indent=4) 
        if pretty else data)


def update_author_json(fn, arr_name, pos, key, value):
    if not os.path.exists(fn):
        print(f"[SKIP] {fn} (not found)")
        return

    try:
        with open(fn, 'r', encoding='utf-8') as read_file:
            data = json.load(read_file)

        data[arr_name][pos][key] = value

        with open(fn, 'w', encoding='utf-8') as write_file:
            json.dump(data, write_file, indent=2)  # ✅ Pretty-print with indentation

        print(f"[SUCCESS] Updated '{key}' to '{value}' at position {pos} in array '{arr_name}' in {fn}")

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"[ERROR] Failed to update JSON: {e}")


# read_print_json('../files_to_read/authors.json', True, True)
update_author_json(
    '../files_to_read/authors.json',
    'authors', 2, 'courses', 6)
