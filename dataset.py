import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, augmentation=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.augmentation = augmentation
        files = os.listdir(self.path)
        # corresponding image and text file have the same names, but different extensions
        self.images = sorted([el for el in files if ".jpg" in el])
        self.labels = sorted([el for el in files if ".txt" in el])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int):
        image_path = os.path.join(self.path, self.images[item])
        label_path = os.path.join(self.path, self.labels[item])
        image = Image.open(image_path).convert("RGB")

        with open(label_path, "r") as f:
            txt_data = f.read().split()
        target = {}
        target["label"] = int(txt_data[0])
        # it is convinient to convert 1 and 2 labels to 0 and 1 labels (0 - cat, 1 - dog)
        target["label"] = np.where(target["label"] == 1, 0.0, 1.0)
        target["boxes"] = [int(el) for el in txt_data[1:]]
        # target['boxes'] = [xmin, ymin, xmax, ymax]
        target = self.relative_coords(image, target)
        # the coordinates of a box are now relative, therefore we can perform transformation:
        if self.transform:
            image = self.transform(image)

        # after resizing we can convert coordinates back to their absolute meanings
        target = self.absolute_cords(image, target)

        if self.augmentation:
            # transform box coordinates to BoundingBoxesOnImage class object to perform augmentation on coordinates
            bb = BoundingBoxesOnImage([BoundingBox(*target["boxes"])], shape=(299, 299))
            image, box = self.augmentation(image=np.array(image), bounding_boxes=bb)
            # convert box coordinates back to a list:
            target["boxes"] = box.items[0].coords.reshape(4).tolist()
            # convert np.array back to pil:
            image = Image.fromarray(image)

        totensor = transforms.ToTensor()
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = totensor(image)
        image = norm(image)

        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["label"] = torch.as_tensor(target["label"], dtype=torch.float32)

        return image, target

    @staticmethod
    def relative_coords(image, target):
        # convert absolute coordinates of a bounding box to relative coordinates
        width, height = image.size
        xmin, ymin = target["boxes"][0] / width, target["boxes"][1] / height
        xmax, ymax = target["boxes"][2] / width, target["boxes"][3] / height
        target["boxes"] = [xmin, ymin, xmax, ymax]
        return target

    def absolute_cords(self, image, target):
        # convert relative coordinates of a bounding box to absolute coordinates
        width, height = image.size
        xmin, ymin = target["boxes"][0] * width, target["boxes"][1] * height
        xmax, ymax = target["boxes"][2] * width, target["boxes"][3] * height
        target["boxes"] = [xmin, ymin, xmax, ymax]
        return target
