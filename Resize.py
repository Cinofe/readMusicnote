import cv2, numpy as np

class Resize:
    def __init__(self):
        self.__src = []

    def __preparing(self):
        try:
            self.__src = cv2.cvtColor(self.__src, cv2.COLOR_RGB2GRAY)
            _, self.__src = cv2.threshold(self.__src, 127, 255, cv2.THRESH_OTSU)
        except Exception as e:
            pass
        finally:
            _, self.__src = cv2.threshold(self.__src, 127, 255, cv2.THRESH_OTSU)


    def resizing(self, src, Scale=2):
        self.__src = src.copy()
        self.__preparing()
        w, h = self.__src.shape
        s = Scale
        dst = np.zeros((w*s, h*s),np.uint8)
        # 원본 이미 0,0 부터 width, height 까지 반복
        for i in range(h):
            for j in range(w):
                # 확대 시킬 스케일 만큼 픽셀 복사, 생성
                for k in range(s):
                    for l in range(s):
                        dst[(j*s-(s-1))+l][(i*s-(s-1))+k] = self.__src[j][i].copy()
        return dst
