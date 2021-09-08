import numpy as np
from PIL import Image
from numpy import newaxis
import os
import cv2
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Datasets(Dataset):

    def __init__(self, path):
        self.path = path

        self.image = os.listdir(os.path.join(path, "image"))
        self.label = os.listdir(os.path.join(path, "label"))
        self.image.sort()
        self.label.sort()
        self.image = self.image
        self.label = self.label
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]

        img_path = [os.path.join(self.path, i) for i in ("image", "label")]
        img = Image.open(os.path.join(img_path[0], image))
        img = np.squeeze(img).astype(np.float32, copy=False)

        img[img < -1000] = -1000
        img[img > 400] = 400
        img = (img + 1000) / 1400
        #img = img[:, :, newaxis]

        label = io.imread(os.path.join(img_path[1], label), as_gray=True, plugin='matplotlib')
        #label.reshape((label.shape[0], label.shape[1], 1))
        label = np.squeeze(label).astype(np.float32, copy=False)
        if np.sum(label) > 0:
            label = label/np.max(label)
        #label = label[:,:,newaxis]


        img = np.expand_dims(img,axis=-1)
        label = np.expand_dims(label,axis=-1)

        img = self.trans(img)
        label = self.trans(label)

        #return self.trans(img), self.trans(label)
        return {'img': img, 'label': label, 'img_name': image}
