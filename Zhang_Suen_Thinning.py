from ctypes import Array
from numpy import array, ndarray
from parent import parent
from multipledispatch import dispatch
import cv2, os
# 0 = WHITE
# 1 = BLACK

WHITE = 255
BLACK = 0

class Thinning(parent):
    def __init__(self):
        self.__src = None
        self.__dst = None
        self.__h = self.__w = None
        self.__counter = 0

    def __Over2_Less6(self,Ps):
        return 2 <= sum(Ps[1:]) <= 6

    def __IO_Pattern(self,Ps):
        counter = 0
        for i in range(1,8):
            if Ps[i] == 0 and Ps[i+1] == 1 :
                counter += 1
        if Ps[-1] == 0 and Ps[0] == 1:
            counter += 1

        if counter == 1:
            return True
        else :
            return False

    def __First_Subiteration(self,Ps):
        '''
        첫 번째 하위 반복 조건
        0. P1 = 검정
        1. 2 <= B(P1) <= 6
        2. A(P1) = 1
        3. P2 * P4 * P6 = 0
        4. P4 * P6 * P8 = 0
        '''
        def C3(Ps):
            return Ps[1] * Ps[3] * Ps[5] == 0

        def C4(Ps):
            return Ps[3] * Ps[5] * Ps[7] == 0

        if Ps[0] == 1 and self.__Over2_Less6(Ps) == True and self.__IO_Pattern(Ps) == 1 and C3(Ps) == True and C4(Ps) == True:
            return True
        else:
            return False

    def __Second_Subiteration(self,Ps):
        '''
        두 번째 하위 반복 조건
        0. P1 = 검정
        1. 2 <= B(P1) <= 6
        2. A(P1) = 1
        3. P2 * P4 * P8 = 0
        4. P2 * P6 * P8 = 0
        '''
        def C3(Ps):
            return Ps[1] * Ps[3] * Ps[7] == 0
        
        def C4(Ps):
            return Ps[1] * Ps[5] * Ps[7] == 0

        if Ps[0] == 1 and self.__Over2_Less6(Ps) == True and self.__IO_Pattern(Ps) == True and C3(Ps) == True and C4(Ps) == True:
            return True
        else:
            return False
    
    def __Matrix(self,i,j):
        P1 = self.__src[j,i]
        P2 = self.__src[j-1,i]
        P3 = self.__src[j-1,i+1]
        P4 = self.__src[j,i+1]
        P5 = self.__src[j+1,i+1]
        P6 = self.__src[j+1,i]
        P7 = self.__src[j+1,i-1]
        P8 = self.__src[j,i-1]
        P9 = self.__src[j-1,i-1]
        return P1,P2,P3,P4,P5,P6,P7,P8,P9
    
    def __digitization(self, img):
        for j in range(self.__h):
            for i in range(self.__w):
                # 검정이면 1로 하양이면 0으로
                if img[j,i] == 0:
                    img[j,i] = 1
                if img[j,i] == 255:
                    img[j,i] = 0
        
        return img

    def __undigitization(self, img):
        for j in range(self.__h):
            for i in range(self.__w):
                # 1이면 검정으로 0이면 하양으로
                if img[j,i] == 0:
                    img[j,i] = 255
                if img[j,i] == 1:
                    img[j,i] = 0
        
        return img
           
    def __Thinning_Algorithm(self,img):
        self.__src = img
        self.__h, self.__w = self.__src.shape
        self.__src = self.__digitization(self.__src)
        self.__dst = self.__src.copy()

        while True:
            for j in range(1, self.__h-1):
                for i in range(1, self.__w-1):
                    Ps = self.__Matrix(i,j)
                    if self.__First_Subiteration(Ps) == True:
                        self.__dst[j,i] = 0
                        self.__counter += 1
            
            self.__src = self.__dst.copy()

            if self.__counter == 0:
                break

            self.__counter = 0

            for j in range(1, self.__h-1):
                for i in range(1, self.__w-1):
                    Ps = self.__Matrix(i,j)
                    if self.__Second_Subiteration(Ps) == True:
                        self.__dst[j,i] = 0
                        self.__counter +=1
            
            self.__src = self.__dst.copy()

            if self.__counter == 0:
                break
            
        self.__src = self.__undigitization(self.__src)
        return self.__src
    @dispatch()
    def Thinning(self):

        imgs = os.listdir(r'thinning_Symbols')
        if len(imgs) != 0:
            for img in imgs:
                os.remove(r'thinning_Symbols/'+img)
        else :
            print('no img')

        imgs = os.listdir(r'Find_Symbols')
        for img in imgs:
            self.__src = cv2.imread(r'Find_Symbols/'+img)
            self.__src = super().GrayScale(self.__src)
            self.__src = super().binary(self.__src)
            self.__dst = self.__Thinning_Algorithm(self.__src)
            cv2.imwrite(r'thinning_Symbols/'+img, self.__dst)

    @dispatch(ndarray)
    def Thinning(self, img):
        self.__src = img
        # self.__src = super().GrayScale(self.__src)
        self.__src = super().binary(self.__src)
        self.__dst = self.__Thinning_Algorithm(self.__src)
    
        return self.__dst

if __name__ == '__main__':
    img = cv2.imread(r'Test_Symbols/143.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

    thinning = Thinning()
    img = thinning.__Thinning_Algorithm(img)
    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.imshow('test',img)
    cv2.waitKey()