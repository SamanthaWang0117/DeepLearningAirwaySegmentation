import imageio as imageio
import nibabel as nib
import nrrd
from PIL import Image
import nibabel as nib
import imageio

#load training data
img = nib.load('/Users/samanthawang/Documents/segmentationvgn/airway/training/15A_source.nii')
nii_data = img.get_fdata()
#print(nii_data.shape)

#load label and save slices
data,header = nrrd.read('/Users/samanthawang/Documents/segmentationvgn/airway/label/15A_segment.seg.nrrd')
#print(data.shape)
for i in range(0,150):
    #iterate slices from the 99th to the 249th
    img_data = (nii_data[:, :, i+100])
    #save slices
    imageio.imwrite("./airway/training/15A/{}.png".format(i),img_data)
    nrrd_image = Image.fromarray(data[:, :, i + 100])
    img = nrrd_image.save("./airway/label/15A/{}.png".format(i))


