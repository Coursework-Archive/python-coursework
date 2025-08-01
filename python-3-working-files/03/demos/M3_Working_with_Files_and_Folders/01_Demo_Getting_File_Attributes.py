import os
from datetime import datetime


def get_file_date(filepath: str) -> str:
    timestamp = os.path.getmtime(filepath)
    dt = datetime.fromtimestamp(timestamp)  # Use fromtimestamp for local time
    return dt.strftime('%d %b %Y %H:%M:%S')


def get_file_attrs(fld):
    with os.scandir(fld) as directory:
        for f in directory:
            if f.is_file():
                print(f'Modified {get_file_date(f.path)} {f.name}')


get_file_attrs('../files/subfolder')
