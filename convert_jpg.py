import matplotlib.image
import os

def convert_dir(dir_name):
    parsed_images = []
    for fname in os.listdir(dir_name):
        im_mat = matplotlib.image.imread('datasets/only_buses/1867.png')[:,:,0:3]
        parsed_images.append(im_mat);
