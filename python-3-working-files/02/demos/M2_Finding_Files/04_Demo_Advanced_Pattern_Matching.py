import os
import fnmatch


def match(fld, search):
    for fn in os.listdir(fld):
        if fnmatch.fnmatch(fn, search):
            print(fn)


file_dir = '../files'

print(f"Searching in {file_dir} for '*_file*.*' ...")
match(file_dir, '*_file*.*')
print(f"Searching in {file_dir} for '*_file_*.*' ...")
match(file_dir, '*_file_*.*')
print(f"Searching in {file_dir} for '*2_*_*.*' ...")
match(file_dir, '*2_*_*.*')
