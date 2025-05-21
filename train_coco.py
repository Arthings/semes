from bibcval4.object_detection import odModule
import typer 
from typing import List, Dict
import os
from torch.utils.data import DataLoader
from dataset import MyVOCDetection
import torch 
import lightning as L9
from pathlib import Path

app= typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    # rootdir = "/home/E097600/data_wsl",
    rootdir: str = "/media/jc_iris/azureml-blobstore-288685e8-7992-4dfa-be46-a46db963aa28/jc_iris",

    # training
    batch_size: int = 16,
    epochs: int = 1,
    lr: float = 1e-3,

):
    rootdir = Path(rootdir)
    datasetdir = rootdir/ "datasets"
    logdir = rootdir / "outputs"/ "voc"


    logdir.mkdir(exist_ok=True, parents=True)
    datasetdir.mkdir(exist_ok=True, parents=True)

    print(f"{logdir=}")
    print(f"{datasetdir=}")

    ds = MyVOCDetection(
        root=datasetdir/ "voc",
        # root = "./logs",
        year="2012",
        image_set="train",
        download=True,
    )

    val_ds = MyVOCDetection(
        root=datasetdir/ "voc",
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
        num_workers=os.cpu_count()//2,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count()//2,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = odModule(pretrained=False)

    trainer = L9.Trainer(
        max_epochs=50,
        default_root_dir=logdir,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_loader, val_loader)

    return 0


if __name__ == "__main__":
    app()