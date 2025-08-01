import os


def ends_with(fld, search):
    for fn in os.listdir(fld):
        if fn.endswith(search):
            print(fn)


def starts_with(fld, search):
    for fn in os.listdir(fld):
        if fn.startswith(search):
            print(fn)


print("Text files found:")
ends_with('../files', '.txt')
print("File(s) that start with '01_test':")
starts_with('../files', '01_test')
