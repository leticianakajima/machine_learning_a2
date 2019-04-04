import numpy as np
import numpy as np
from sklearn.cluster import KMeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters
        Parameters
        ----------
        img : a (H,W,3) numpy array
        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices
        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour
        """

        H, W, _ = img.shape
        img_new = np.reshape(img, [H * W, 3])
        model = KMeans(n_clusters=2**self.b, n_init=3)

        model.fit(img_new)
        y = model.predict(img_new)

        self.colours = model.cluster_centers_

        #store the cluster means and return cluster assignment
        quantized_img = np.reshape(y, (H, W))
        return quantized_img


    def dequantize(self, quantized_img):
        H, W = quantized_img.shape
        img = np.zeros((H,W,3), dtype='uint8')

        for h in range (0,H):
            for w in range (0,W):
                img[h,w] = quantized_img[h,w]
                img[h,w] = self.colours[quantized_img[h,w]]
                #value of the quantized image = index of the colour scheme

        return img
