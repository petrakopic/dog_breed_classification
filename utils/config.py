from pathlib import Path
import yaml


class ModelConfig:
    def __init__(self, config_file_path: str):
        super().__init__()
        with open(f"{Path(__file__).parent.parent}/{config_file_path}", "r") as f:
            self.values = yaml.safe_load(f)

    def get_value(self, name):
        if self.values.get(name) is not None:
            return self.values[name]
        return None

    def to_dict(self):
        return self.__dict__()

    def __dict__(self):
        return self.values

    def print(self):
        print("Model configurations:\n")
        print(self.values)
