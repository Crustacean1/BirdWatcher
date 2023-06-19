import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class BirdWatcher(pl.LightningModule):
    def __init__(self, arch="UNET", encoder_name="resnet34", in_channels=3, out_classes=1):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes)

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(
            params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(
            params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou
        }
        self.log_dict(metrics, prog_bar=True)
    # Metrics
    # Training

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    # Validation

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    # Test

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
