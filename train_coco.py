from training import odModule
import typer
from typing import List, Dict
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataset import MyVOCDetection
import torch
import lightning as L9
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from strategies import random_selection, entropy_selection
from copy import deepcopy
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)

strat_dict = {
    "random": random_selection,
    "entropy": entropy_selection,
}


torch.set_float32_matmul_precision("high")


@app.command()
def main(
    # rootdir = "/home/E097600/data_wsl",
    rootdir: str = "./logs",
    # training
    batch_size: int = 32,
    epochs: int = 20,
    workers: int = 0,
    # AL
    reps: int = 1,
    strat: str = "random",
):
    rootdir = Path(rootdir)
    datasetdir = rootdir / "datasets"
    logdir = rootdir / "outputs" / "voc"
    np.random.seed(38)
    logdir.mkdir(exist_ok=True, parents=True)
    datasetdir.mkdir(exist_ok=True, parents=True)

    print(f"{logdir=}")
    print(f"{datasetdir=}")

    ds = MyVOCDetection(
        root=datasetdir / "voc",
        # root = "./logs",
        year="2012",
        image_set="train",
        download=True,
    )

    val_ds = MyVOCDetection(
        root=datasetdir / "voc",
        # root = "./logs",
        year="2012",
        image_set="val",
        download=True,
    )

    def collate_fn(batch):
        data = torch.stack([x["image"] for x in batch])
        targets = []
        for x in batch:
            target = {}
            target["boxes"] = x["boxes"]
            target["labels"] = x["labels"]

            targets.append(target)

        return data, targets

    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2 * batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    def create_model():
        return odModule(pretrained=True)

    def create_trainer(cycle):
        return L9.Trainer(
            max_epochs=epochs,
            default_root_dir=logdir,
            check_val_every_n_epoch=epochs,
            log_every_n_steps=50,
            num_sanity_val_steps=2,
            enable_checkpointing=True,
            logger=CSVLogger(logdir, name=logdir / strat / f"Training_{cycle}_{rep}"),
        )

    # Start of Selection
    full_indices = torch.arange(len(ds))

    with open(f"metrics_{strat}.txt", "w") as f:
        pass
    for rep in range(reps):
        model = create_model()

        # initial selection

        remaining_indices = full_indices
        selected_indices = strat_dict[strat](
            model,
            base_ds=ds,
            unlabeled_indices=remaining_indices,
            budget=int(0.1 * len(ds)),
        )
        train_indices = selected_indices
        remaining_indices = np.setdiff1d(full_indices, train_indices)

        # Training
        model = create_model()
        trainer = create_trainer(0)
        training_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=RandomSampler(train_indices),
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        trainer.fit(model, training_loader, val_loader)
        trainer.test(model, val_loader)

        metrics_to_log: dict = trainer.logged_metrics
        AL_params = {
            "cycle": 0,
            "rep": rep,
        }

        metrics_to_log.update(AL_params)
        with open(f"./logs/metrics_{strat}.txt", "a") as f:
            for key, values in metrics_to_log.items():
                f.write(f"{str(values.item())};")
            f.write("\n")

        m0_weights = deepcopy(model.state_dict())

        ordered_items = strat_dict[strat](
            model,
            base_ds=ds,
            unlabeled_indices=remaining_indices,
            budget=len(remaining_indices),
        )

        # 75 80 85
        items_to_add = [3716, 4002, 4288]
        for cycle, number_items in enumerate(items_to_add):
            selected_indices = ordered_items[-number_items:]

            # Training
            model = create_model()
            model.load_state_dict(m0_weights)
            trainer = create_trainer(cycle)
            training_loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=RandomSampler(
                    np.concatenate((train_indices, selected_indices))
                ),
                num_workers=workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            trainer.fit(model, training_loader)
            trainer.test(model, val_loader)

            metrics_to_log: dict = trainer.logged_metrics
            AL_params = {
                "cycle": cycle,
                "rep": rep,
            }

            metrics_to_log.update(AL_params)
            with open(f"./logs/metrics_{strat}.txt", "a") as f:
                for key, values in metrics_to_log.items():
                    f.write(f"{str(values.item())};")
                f.write("\n")
    return 0


if __name__ == "__main__":
    app()
