import importlib
from skimage.transform import resize
import matplotlib.pyplot as plt
import data.mpii.mpii_data_handler as data_handler
import data.mpii.data_provider as data_provider
from utils.transparent_imshow import transp_imshow

# creating training data
data_handler.init()
train, valid = data_handler.setup_val_split()
data = [train, valid]
# get configurations
config = importlib.import_module('utils.config').__config__
data_provider.init(config)
# get image and heat map
ds = data_provider.Dataset(config=config, ds=data_handler, index=data)
img, heat_map = ds.loadImage(1729)  # input image index
plt.imshow(img)
for i in range(heat_map.shape[0]):
    hm = heat_map[i, :, :]
    hm = resize(hm, (256, 256), anti_aliasing=True)
    transp_imshow(hm, cmap='hsv')   # custom function
plt.show()