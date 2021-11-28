from dataset import Dataset
from train_fuctions import training, validation
from net import PretrainedNet
import torch
import torchvision.transforms as transforms
import numpy as np
import imgaug

np.random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.backends.cudnn.deterministic = True

augmentations = imgaug.augmenters.Sequential(
    [imgaug.augmenters.flip.Fliplr(0.5),
     imgaug.augmenters.flip.Flipud(0.5)]
)
train_path = '/home/wildkatze/cats_dogs_dataset/train/'
test_path = '/home/wildkatze/cats_dogs_dataset/valid/'
resize = transforms.Compose([transforms.Resize((299, 299))])

train_data = Dataset(train_path, resize, augmentation=augmentations)
test_data = Dataset(test_path, resize)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PretrainedNet()
net = training(net=net, n_epoch=15, lr=0.01, dataloader=train_dataloader,  device=device,
               transfer_learning=True)

validation(net, test_dataloader, device)
