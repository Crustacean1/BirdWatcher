import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from loader import Loader
from bird_watcher import BirdWatcher
import matplotlib.pyplot as plt

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


def evaluate(model, test_dataloader):
    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
        predictions = torch.argmax(
            torch.nn.functional.softmax(logits, dim=1), dim=1)
        for image, ground_truth,  prediction in zip(batch["image"], batch["mask"], predictions):
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth.numpy().argmax(axis=0))
            plt.title("Truth")
            plt.subplot(1, 3, 3)
            plt.imshow(prediction.numpy().squeeze())
            plt.title("Prediction")
            plt.show()


training_data = Loader("./PetsClassified", train=True)
test_data = Loader("./PetsClassified", train=False)

print(
    f"Starting training with {len(training_data)} training images and {len(test_data)} testing images")

for i in range(8):
    sample = training_data[i]
    image = sample["image"]
    mask = sample["mask"]
    print(f"shapes: {image.shape} {mask.shape}")

    plt.subplot(1, 2, 1)
    plt.imshow(image.transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.argmax(axis=0))

    plt.show()


batch_size = 8

train_dataloader = DataLoader(
    training_data, batch_size=batch_size, num_workers=4, shuffle=True)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, num_workers=4, shuffle=False)

model = BirdWatcher("UNET", "resnet34", in_channels=3, out_classes=4)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=5)

torch.save(model.state_dict(), "model_weights.pth")

trainer.fit(model, train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader)

evaluate(model, test_dataloader)
