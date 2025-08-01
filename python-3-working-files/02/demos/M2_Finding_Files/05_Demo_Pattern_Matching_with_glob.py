from pathlib import Path


def glob_match(fld, search):
    p = Path(fld)
    for n in p.glob(search):
        print(n)


print("Searching files/ for '*2*.t*' ...")
glob_match('../files', '*2*.t*')
print("Searching subfolder/ for '*1_*_*.t*' ...")
glob_match('../files/subfolder', '*1_*_*.t*')
