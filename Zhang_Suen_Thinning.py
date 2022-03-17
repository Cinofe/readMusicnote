import cv2
# 0 = WHITE
# 1 = BLACK

WHITE = 255
BLACK = 0

class Thinning:
    def __init__(self):
        self.__src = None
        self.__dst = None
        self.__h = self.__w = None
        self.__counter = 1

    def __Over2_Less6(self,Ps):
        return 2 <= sum(Ps[1:]) <= 6

    def __IO_Pattern(self,Ps):
        counter = 0
        for i in range(1,8):
            if Ps[i] == 0 and Ps[i+1] == 1:
                counter += 1

        if counter == 1:
            return True
        else :
            return False

    def __First_Subiteration(self,Ps):
        '''
        첫 번째 하위 반복 조건
        1. 2 <= B(P1) <= 6
        2. A(P1) = 1
        3. P2 * P4 * P6 = 0
        4. P4 * P6 * P8 = 0
        '''
        def C3(Ps):
            _,P2,_,P4,_,P6,_,_,_ = Ps
            return P2 * P4 * P6 == 0

        def C4(Ps):
            _,_,_,P4,_,P6,_,P8,_ = Ps
            return P4 * P6 * P8 == 0

        if self.__Over2_Less6(Ps) == True and self.__IO_Pattern(Ps) == 1 and C3(Ps) == True and C4(Ps) == True:
            return True
        else:
            return False


    def __Second_Subiteration(self,Ps):
        '''
        두 번째 하위 반복 조건
        1. 2 <= B(P1) <= 6
        2. A(P1) = 1
        3. P2 * P4 * P8 = 0
        4. P2 * P6 * P8 = 0
        '''
        def C3(Ps):
            _,P2,_,P4,_,_,_,P8,_ = Ps
            return P2 * P4 * P8 == 0
        
        def C4(Ps):
            _,P2,_,_,_,P6,_,P8,_ = Ps
            return P2 * P6 * P8 == 0

        if self.__Over2_Less6(Ps) == True and self.__IO_Pattern(Ps) == True and C3(Ps) == True and C4(Ps) == True:
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
        h,w = img.shape
        for i in range(w):
            for j in range(h):
                # 검정이면 1로 하양이면 0으로
                if img[j,i] == 0:
                    img[j,i] = 1
                if img[j,i] == 255:
                    img[j,i] = 0
        
        return img
        
        
    def Thinning(self,img):
        self.__src = img
        self.__dst = self.__src.copy()
        self.__src = self.__digitization(self.__src)
        self.__h, self.__w = self.__src.shape
        
        k = 0
        while True:
            print(f'iteration = {k}')
            for i in range(1, self.__w-1):
                for j in range(1, self.__h-1):
                    Ps = self.__Matrix(i,j)
                    if self.__First_Subiteration(Ps) == True:
                        self.__dst[j,i] = 255
                        self.__counter += 1
            if self.__counter == 0:
                break
            self.__counter = 0

            for i in range(1, self.__w-1):
                for j in range(1, self.__h-1):
                    Ps = self.__Matrix(i,j)
                    if self.__Second_Subiteration(Ps) == True:
                        self.__dst[j,i] = 255
                        self.__counter +=1

            if self.__counter == 0:
                break
            k += 1
        
        return self.__dst
## 무한 반복 발생.
if __name__ == '__main__':
    img = cv2.imread(r'Test_Symbols/143.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

    thinning = Thinning()
    img = thinning.Thinning(img)
    cv2.imshow('test',img)
    cv2.waitKey()