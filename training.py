import lightning as L
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import torchmetrics
from typing import List, Dict
import cv2
from torchvision.ops import nms
from omegaconf import OmegaConf


def od_nms(predictions:List[Dict[str, torch.Tensor]], iou_threshold:float = 0.5) -> List[Dict[str, torch.Tensor]]:
    """Apply NMS to the predictions

    Args:
        predictions (List[Dict[str, torch.Tensor]]): List of prediciton, one prediction refer to the predictions of one image
        iou_threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        List[Dict[str, torch.Tensor]]: _description_
    """

    prediction_after_nms = []
    for i in range(len(predictions)):
        boxes = predictions[i]["boxes"]
        scores = predictions[i]["scores"]
        labels = predictions[i]["labels"]

        keep = nms(boxes, scores, iou_threshold)
        
        predictions[i]["boxes"] = boxes[keep]
        predictions[i]["scores"] = scores[keep]
        predictions[i]["labels"] = labels[keep]

    return predictions



class odModule(L.LightningModule):
    def __init__(self, config=None, pretrained = True):
        super().__init__()
        # if config.checkpoint is not None:
        # print(f"checkpoint from {config.checkpoint}")
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained = pretrained, num_classes=21)

        self.config = OmegaConf.create({
            "lr": 1e-5,
            "batch_size": 2,
            "world_size": 1,
            "epochs": 10,
        })

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.detection.mean_ap.MeanAveragePrecision(
                    # extended_summary=True, 
                    class_metrics=True, 
                    iou_type="bbox",
                ),
            ]
        )   
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")
    
        self.save_hyperparameters(ignore=["train_ds"])

    @staticmethod
    def prepare_batch(batch):
        images, targets = batch
        return images, targets

    def forward(self, x, y=None):
        if y is not None:
            return self.model(x, y)
        else:
            preds = self.model(x)
            return preds

    def predict(self, x):
        """Forward the model then run NMS (for evaluation)

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        preds = self.model(x)
        # preds_nms = self.nms.apply_batch(preds)
        return preds

    def forward_proba(self, x):
        """used in AL functions, in Instance segmentation == forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return self.forward(x)

    def forward_embedding(self, x):
        """used in Core-Set algorithms the create the embedding matrix

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        bs = len(x)
        images = torch.stack(x)
        return self.model.backbone(images)["pool"].view(bs, -1)

    def training_step(self, batch, batch_idx):
        img_b, target_b = self.prepare_batch(batch)
        bs = len(img_b)

        loss_dict = self.model(img_b, target_b)
        loss_dict["loss_total"] = sum(loss_dict.values())

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
            prog_bar=True,
        )

        return {"loss": loss_dict["loss_total"]}

    def validation_step(self, batch, batch_idx):
        img_b, target_b = self.prepare_batch(batch)
        output_nms = self.predict(img_b)

        self.val_metrics(output_nms, target_b)
        return

    def on_validation_epoch_end(self):
        
        m = self.val_metrics.compute()
        m = {i:j for i,j in m.items() if j.nelement() ==1}
        
        self.log_dict(m, on_epoch=True, sync_dist=False)

        self.val_metrics.reset()
        return

    def test_step(self, batch, batch_idx):
        img_b, target_b = self.prepare_batch(batch)

        output_nms = self.predict(img_b)

        self.test_metrics(output_nms, target_b)
        return

    def on_test_epoch_end(self):
        m = self.test_metrics.compute()
        m = {i:j for i,j in m.items() if j.nelement() ==1}

        self.log_dict(m, on_epoch=True, sync_dist=False)

        self.test_metrics.reset()

        return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4 * self.config.batch_size / 16,
        )

        scheduler_nsteps = self.config.epochs * self.config.world_size
        # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=scheduler_nsteps, eta_min=self.config.lr / 10
        # )

        # sched_config1 = {"scheduler": scheduler1, "interval": "epoch"}


        return [optimizer]#, [sched_config1]

if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 10,
        "world_size": 1,
        "checkpoint": None
    })
    model = odModule(config, pretrained=False)
    model.eval()

    model(torch.rand(4,3,512,512))

