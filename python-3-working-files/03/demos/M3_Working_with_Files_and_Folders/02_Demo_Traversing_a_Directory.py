import os


def traverse(fld):
    for fld_path, dirs, fls in os.walk(fld):
        print(f'Folder: {fld_path}')
        for fn in fls:
            print(f'\t{fn}')


traverse('../files')
