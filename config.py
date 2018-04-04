import uuid
from os import listdir, path, mkdir, walk, makedirs
from datetime import datetime
from helper.config_helper import get_savedir

from util import assert_dir, yaml_dump_log
from shutil import copy2, rmtree


def get_config():
    config = {
        "general":
            {
                "name": "",
                "date": datetime.now().strftime("%Y/%m/%d_%H:%M:%S"),
                "hash": str(uuid.uuid4())[0:8]
            },
        "device": [0],  # 0, if using GPUs
        "dataset": {
            "module": "dataset",  # == dataset.py
            "class": "NibDataset",
            "train": {
                "params": {
                    "directory": "/home/hashimoto/data/train"
                },
                "file": None
            },
            "valid": {
                "params": {
                    "directory": "/home/hashimoto/data/valid"
                },
                "file": None
            }
        },
        "model": {
            "module": "model",  # == model.py
            "class": "ExampleModel",
            "params": {
                "c": 12  # to be modified according to your model
            }
        },
        "optimizer": {
            "module": "chainer.optimizers",
            "class": "Adam",
            "params": {
            },
            "hook":
                [
                    {
                        "module": "chainer.optimizer",
                        "class": "WeightDecay",
                        "params": {
                            "rate": 0.0001
                        }
                    }
                ]
        },
        "trainer": {
            "params": {
                "stop_trigger": [1000, "epoch"],
                "out": None
            },
            "model_interval": [1, "epoch"],
            "log_interval": [100, "iteration"],
            "eval_interval": [1, "epoch"]
        },
        "batch": {
            "train": 1,
            "valid": 1
        },
        "save": {
            "root": "/home/hashimoto/out",
            "directory": None,
            "program": {
                "directory": None
            },
            "log": {
                "directory": None,
                "file": "config.yml"
            },
            "model": {
                "directory": None
            }
        }
    }

    try:
        assert_dir(config["dataset"]["train"]["params"]["directory"])
        assert_dir(config["dataset"]["valid"]["params"]["directory"])
        assert_dir(config["save"]["root"])
    except FileNotFoundError as e:
        print(e)
        exit(1)

    config["dataset"]["train"]["file"] = sorted(listdir(config["dataset"]["train"]["params"]["directory"]))
    config["dataset"]["valid"]["file"] = sorted(listdir(config["dataset"]["valid"]["params"]["directory"]))

    config["general"]["name"] = get_savedir(config["save"]["root"], config["general"])

    config["save"]["directory"] = path.join(config["save"]["root"], config["general"]["name"])
    mkdir(config["save"]["directory"])
    config["save"]["program"]["directory"] = path.join(config["save"]["directory"], "program")
    mkdir(config["save"]["program"]["directory"])
    config["save"]["log"]["directory"] = path.join(config["save"]["directory"], "log")
    mkdir(config["save"]["log"]["directory"])
    config["save"]["model"]["directory"], config["trainer"]["params"]["out"] = [path.join(config["save"]["directory"], "model")] * 2
    mkdir(config["save"]["model"]["directory"])

    for root, dirs, files in walk("."):
        for file in files:
            body, ext = path.splitext(file)
            if ext == ".py":
                src = path.join(root, file)
                dest = path.join(config["save"]["program"]["directory"], path.join(root, file))
                try:
                    makedirs(path.dirname(dest))
                except FileExistsError:
                    pass
                copy2(src, dest)
    with open(path.join(config["save"]["log"]["directory"], config["save"]["log"]["file"]), "a"):
        print(yaml_dump_log(config))

    return config


def destroy_config(config):
    rmtree(config["save"]["directory"])
    exit(0)

if __name__ == "__main__":
    c = get_config()
    print(yaml_dump_log(c))
    destroy_config(c)
