import os
import datetime
import time
from datetime import timedelta

import pandas as pd


def ls(directory="input"):
    files = []
    for file in sorted(os.scandir(directory), key=lambda file: file.name):
        files.append({
            "name": file.name,
            "filesize (MB)": round(file.stat().st_size / 1024 / 1024, 2),
            "last modified": datetime.datetime.fromtimestamp(file.stat().st_mtime)
    })
    output = pd.DataFrame(files)
    with pd.option_context('display.max_colwidth', None):
        display(output)
    print(f"Total: {output['filesize (MB)'].sum().round()}MB")


class Timer:
    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def report(self, message, reset=True):
        print(message)
        now = time.time()
        print(now - self.last, 'seconds')
        if reset:
            self.last = now
        print()

    def report_total(self):
        end = time.time()
        elapsed = end - self.start
        print('total execution time:', elapsed)
        print(str(timedelta(seconds=elapsed)))
