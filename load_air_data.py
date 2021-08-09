import nibabel as nib
import nrrd
from PIL import Image

#load training data
train_img,header = nib.load('/Users/samanthawang/Documents/segmentationvgn/airway/training/18A_source.nii')
print(train_img)

#load label and save slices
data,header = nrrd.read('/Users/samanthawang/Documents/segmentationvgn/airway/label/15A_segment.seg.nrrd')
print(data.shape) # z,y,x [n,w,h] n, y=宽，x=高
for i in range(0,50):
    nrrd_image = Image.fromarray(data[:, :, i+100])
    img = nrrd_image.save("./airway/label/15A/{}.png".format(i))

