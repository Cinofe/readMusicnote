import cv2, numpy as np, math
## 현재 쓸모 없지만 남겨둠
class Resize:
    def __init__(self):
        self.__src = []

    def resize(self, src, size = None):
        self.__src = src.copy()
        h, w = self.__src.shape

        if size == None:
            size = [h,w]
        
        dst = np.zeros((size[0], size[1]),np.uint8)

        x_scale = h/dst.shape[0]
        y_scale = w/dst.shape[1]

        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                xp, yp = math.floor(x*x_scale), math.floor(y*y_scale)
                dst[x,y] = self.__src[xp,yp]
        return dst
