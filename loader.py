import os
from torch.utils.data import Dataset
import cv2
import numpy as np


class Loader(Dataset):
    def __init__(self, root, train):
        self.root = root
        self.train = train

        image_files = filter(
            lambda filepath: "image" in filepath, os.listdir(self.root))
        image_count = sum(1 for _ in image_files)

        training_count = int(image_count * 0.9)
        testing_count = image_count - training_count

        (start, count) = (0, training_count) if train else (
            training_count, testing_count)
        self.start = start
        self.count = count

    def __getitem__(self, idx):
        image_id = self.start + idx
        image_path = os.path.join(self.root, f"image.{image_id}.png")
        mask_path = os.path.join(self.root, f"mask.{image_id}.png")

        image = load_image(image_path, cv2.COLOR_BGR2RGB)
        mask = load_image(mask_path, cv2.COLOR_BGR2GRAY)

        image = image.transpose(2, 0, 1)

        c1 = np.copy(mask)
        c1[c1 != 0.0] = 0.0

        c2 = np.copy(mask)
        c2[c2 != 1.0] = 0.0
        c2[c2 == 1.0] = 1.0

        c3 = np.copy(mask)
        c3[c3 != 2.0] = 0.0
        c3[c3 == 2.0] = 1.0

        c4 = np.copy(mask)
        c4[c4 != 3.0] = 0.0
        c4[c4 == 3.0] = 1.0

        mask = np.dstack((c1, c2, c3, c4)).transpose(2, 0, 1)

        return {"image": image, "mask": mask}

    def __len__(self):
        return self.count


def load_image(path, color):
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(256, 256))
    image = cv2.cvtColor(image, color)
    return image
