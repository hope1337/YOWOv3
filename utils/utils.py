import json
import os

def save_dict_json(dict2save, path2save):
    with open(path2save, "w") as f:
            json.dump(dict2save, f, indent=4)

def load_dict_json(path):
    with open(path, "r") as f:
        load_dict = json.load(f)
    return load_dict