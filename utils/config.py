from pathlib import Path

import yaml


class Config(dict):
    def __init__(self, config_file_path:str):
        super().__init__()
        with open(f"{Path(__file__).parent}/{config_file_path}", 'r') as f:
            self._yaml = yaml.safe_load(f)

    def __getattr__(self, name):
        if self._yaml.get(name) is not None:
            return self._yaml[name]
        return None

    def print(self):
        print('Model configurations:\n')
        print(self._yaml)


