import struct
from numpy import *
import numpy as np

#function to read and load  training and test label
#The below method was referred from stackoverflow
def parse_labels(fileName):
    """Parse labels from the binary file."""
    filePath = "./dataset/" + fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()
    magic, n = struct.unpack_from('>2i', data)
    assert magic == 2049, "Wrong magic number: %d" % magic

    labels = struct.unpack_from('>%dB' % n, data, offset=8)
    return labels

#function to read and load  training and test image files
#The below method was referred from stackoverflow
def parse_images(fileName):
    """Parse images from the binary file."""
    filePath = "./dataset/" + fileName
    with open(filePath, "rb") as binary_file:
        data = binary_file.read()

    magic, n, rows, cols = struct.unpack_from('>4i', data)
    assert magic == 2051, "Wrong magic number: %d" % magic

    num_pixels = n * rows * cols
    pixels = struct.unpack_from('>%dB' % num_pixels, data, offset=16)

    pixels = asarray(pixels, dtype=np.float)

    images = pixels.reshape((n, cols, rows))
    return images
