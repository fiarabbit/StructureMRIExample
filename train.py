from importlib import import_module

from config import get_config, destroy_config
from util import yaml_dump

from chainer.iterators import SerialIterator as Iterator
from chainer.training.updater import StandardUpdater as Updater
from chainer.training import Trainer
from chainer.training.extensions import snapshot_object, observe_lr, LogReport, PrintReport, Evaluator, ProgressBar

def run():
    config = get_config()
    print(yaml_dump(config))
    s = ""
    while not (s == "y" or s == "n"):
        s = input("ok? (y/n): ")
        if s == "n":
            destroy_config(config)

    device = config["device"][0] if isinstance(config["device"], list) else config["device"]

    Model = getattr(import_module(config["model"]["module"]), config["model"]["class"])
    model = Model(**config["model"]["params"])

    Dataset = getattr(import_module(config["dataset"]["module"]), config["dataset"]["class"])
    train_dataset = Dataset(**config["dataset"]["train"]["params"])
    valid_dataset = Dataset(**config["dataset"]["valid"]["params"])

    train_iterator = Iterator(train_dataset, config["batch"]["train"], True, True)
    valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False)

    Optimizer = getattr(import_module(config["optimizer"]["module"]), config["optimizer"]["class"])
    optimizer = Optimizer(**config["optimizer"]["params"])
    optimizer.setup(model)
    for hook_config in config["optimizer"]["hook"]:
        Hook = getattr(import_module(hook_config["module"]), hook_config["class"])
        hook = Hook(**hook_config["params"])
        optimizer.add_hook(hook)

    updater = Updater(train_iterator, optimizer, device=device)

    trainer = Trainer(updater, **config["trainer"]["params"])
    trainer.extend(snapshot_object(model, "model_iter_{.updater.iteration}"), trigger=config["trainer"]["model_interval"])
    trainer.extend(observe_lr(), trigger=config["trainer"]["log_interval"])
    trainer.extend(LogReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"], trigger=config["trainer"]["log_interval"]))
    trainer.extend(PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"]), trigger=config["trainer"]["log_interval"])
    trainer.extend(Evaluator(valid_iterator, model, device=device), trigger=config["trainer"]["eval_interval"])
    trainer.extend(ProgressBar(update_interval=10))

    print("start training")
    trainer.run()


if __name__ == "__main__":
    run()
