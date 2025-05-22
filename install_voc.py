import torchvision
from typer import Typer
from pathlib import Path

app= Typer(pretty_exceptions_enable=False)

@app.command()
def main(rootdir:str = "./logs"):
    rootdir = Path(rootdir)
    print(rootdir)
    ds = torchvision.datasets.VOCDetection(
        root= rootdir / "voc",
        year="2012",
        image_set="train",
        download=True,
    )
    ds = torchvision.datasets.VOCDetection(
        root= rootdir / "voc",
        year="2012",
        image_set="val",
        download=True,
    )

if __name__ == "__main__":
    app()