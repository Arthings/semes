import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCDetection
import numpy as np
from typing import Dict, Optional
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf


def boxdict_to_boxtens(bndbox:Dict)-> torch.tensor:
    xmin = float(bndbox["xmin"])
    ymin = float(bndbox["ymin"])
    xmax = float(bndbox["xmax"])
    ymax = float(bndbox["ymax"])
    return torch.Tensor([xmin, ymin, xmax, ymax])


class MyVOCDetection(VOCDetection):
    def __init__(self, *args, **kwargs):
        super(MyVOCDetection, self).__init__(*args, **kwargs)
        categories = OmegaConf.load("voc_cats.yaml")["names"]
        self.categories = [cat for i, cat in categories.items()]

        self.train_transform = A.Compose(
            [
                A.Resize(640,640),
                A.HorizontalFlip(),
                A.Normalize(normalization="min_max"),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        self.val_transform = A.Compose(
            [
                A.Resize(640,640),
                A.Normalize(normalization="min_max"),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    def __getitem__(self, index):
        img, target = super(MyVOCDetection, self).__getitem__(index)

        if self.image_set == "train":
            transformed = self.train_transform(
                image=np.array(img),
                bboxes=torch.stack([boxdict_to_boxtens(obj["bndbox"]) for obj in target["annotation"]["object"]]),
                labels=[self.categories.index(obj["name"]) for obj in target["annotation"]["object"]],
            )
        elif self.image_set == "val":
            transformed = self.val_transform(
                image=np.array(img),
                bboxes=torch.stack([boxdict_to_boxtens(obj["bndbox"]) for obj in target["annotation"]["object"]]),
                labels=[self.categories.index(obj["name"]) for obj in target["annotation"]["object"]],
            )
        
        transformed["boxes"] = torch.from_numpy(transformed["bboxes"])
        transformed.pop("bboxes")
        transformed["original_image"] = img
        transformed["image"] = transformed["image"].float()
        transformed["labels"] = torch.Tensor(transformed["labels"]).to(torch.int64)
        return transformed


    def show_item(self, idx:Optional[int] = None) -> None:

        if idx is None:
            idx = np.random.randint(0, len(self))
            print(f"idx = {idx}")
        detect_item = self[idx]
        original_image = detect_item["original_image"]
        image = detect_item["image"]
        
        with_boxes = draw_bounding_boxes(
            # image=torch.from_numpy(np.array(original_image)).permute(2, 0, 1),
            image = image,
            boxes=detect_item["boxes"],
            labels=[self.categories[int(i)] for i in detect_item["labels"]],
        )
        plt.imshow(with_boxes.permute(1, 2, 0).numpy())
        plt.axis("off")


if __name__ == "__main__":
    ds = MyVOCDetection(
        root="/home/E097600/data_wsl/voc",
        year="2012",
        image_set="train",
        download=False,
    )

    ds[0]
    ds.show_item()    
    2+2
