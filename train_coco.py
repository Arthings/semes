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
    epochs: int = 30,
    workers:int = 0,

    # AL
    reps: int = 1,
    strat:str = "random",
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
        batch_size=2*batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


    def create_model():
        return odModule(pretrained=False)

    def create_trainer(cycle):
        return L9.Trainer(
            max_epochs=epochs,
            default_root_dir=logdir,
            check_val_every_n_epoch=epochs,
            log_every_n_steps=50,
            num_sanity_val_steps=2,
            enable_checkpointing=True,
            logger=CSVLogger(logdir, name=f"Training_{cycle}_{rep}"),
        )

    # Start of Selection
    full_indices = torch.arange(len(ds))

    with open(f"metrics_{strat}.txt", "w") as f:
        pass
    for rep in range(reps):
        items_to_add = [2858, 572, 1143, 1144]  # 50% 60% 80% 100%
        model = create_model()

        for cycle, number_items in enumerate(items_to_add):
            if cycle == 0:
                # Random selection for the first cycle
                train_indices = np.random.choice(
                    full_indices, number_items, replace=False
                )
                remaining_indices = np.setdiff1d(full_indices, train_indices)
            else:
                selected_indices = strat_dict[strat](model, base_ds=ds, unlabeled_indices=remaining_indices, budget=number_items)
                train_indices = np.concatenate((train_indices, selected_indices))
                remaining_indices = np.setdiff1d(full_indices, train_indices)

            # Training
            model = create_model()
            trainer = create_trainer(cycle)
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
            metrics_to_log : dict = trainer.logged_metrics

            with open(f"metrics_{strat}.txt", "a") as f:
                for key, values in metrics_to_log.items():
                    f.write(f"{str(values)};")
                f.write("\n")


            
    return 0


if __name__ == "__main__":
    app()
