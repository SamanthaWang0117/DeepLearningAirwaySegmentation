import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image



class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.name1)

    def __trans__(self, img, size):

        h, w = img.shape[0:2]

        _w = _h = size

        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left

        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):

        name1 = self.name1[index]
        name2 = self.name2[index]

        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]

        img_o = cv2.imread(os.path.join(img_path[0], name1))
        _, img_l = cv2.VideoCapture(os.path.join(img_path[1], name2)).read()
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        img_o = self.__trans__(img_o, 352)
        img_l = self.__trans__(img_l, 352)

        return self.trans(img_o), self.trans(img_l)



data = r"C:\software\workstation\project\wss\DRIVE_2\training"

train_data = Datasets(path=data)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)


# i = 1
for i, (a,b) in enumerate(train_loader):
    # print(i)
    # print(a.shape)
    # print(b.shape)
    save_image(a, r"C:\software\workstation\project\wss\result\{}.gif".format(i), nrow=1)
    save_image(b, r"C:\software\workstation\project\wss\result\{}.tif".format(i), nrow=1)
    # i += 1
