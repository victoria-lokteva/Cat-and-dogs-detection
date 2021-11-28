import pytest
from dataset import Dataset
import numpy as np
from PIL import Image

path = '/home/wildkatze/cats_dogs_dataset/'

image_list = ["/american_bulldog_128.jpg", "/leonberger_180.jpg", "/Maine_Coon_217.jpg", "/newfoundland_141.jpg",
              "/Persian_144.jpg",  "/pug_104.jpg", "/Russian_Blue_112.jpg", "/pomeranian_160.jpg", "/pomeranian_13.jpg"]

names = [name[1:-4] + ".npy" for name in image_list]


def test_relcoords():
    d = Dataset(path)
    for name in names:
        image = np.load(path + name)
        file = path+ name[:-3] + 'txt'
        target = {}
        with open(file, 'r') as f:
            target['boxes'] = f.readline()[1:].split()
        target['boxes'] = list(map(int, target['boxes'] ))
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        target = d.relative_coords(image, target)

        for i in range(3):
            assert target['boxes'][i] <= 1
            assert  target['boxes'][i] >= 0

test_relcoords()
