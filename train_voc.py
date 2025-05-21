from ultralytics import YOLO, settings
import torchvision
import typer 
from typing import List, Dict
import os
from dataset import MyVOCDetection
app= typer.Typer(pretty_exceptions_enable=False)

def yolo_output_to_normal(yoloOutput:List)->List[Dict]:

    ret = []
    for output in yoloOutput:
        ret.append({
            "boxes": output.boxes.xyxy,
            "scores" : output.boxes.conf,
            "label" : output.boxes.cls,
        })
    return ret
    
@app.command()
def main(
    datasetroot = "/home/E097600/data_wsl/voc",
    logdir: str = "/home/E097600/semes_oui/datasets",
):

    print(f"{logdir=}")

    print(settings)
    settings.update({"datasets_dir": "/home/E097600/semes_oui/datasets"})
    print(settings)
    logdir = os.path.join(logdir, "voc")
    os.makedirs(
        logdir, 
        exist_ok=True
    )

    # Load a model
    model = YOLO(task = "detect")

    model.train(data = "voc_cats.yaml", epochs = 1)

    
    return 0


if __name__ == "__main__":
    app()