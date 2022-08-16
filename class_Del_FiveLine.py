from parent import parent
from multipledispatch import dispatch
import cv2, numpy as np

class Del_FiveLine(parent):
    '''
    대부분의 악보들의 오선 시작 위치는 악보의 너비 약 5% 지점에서 시작
    악보의 수직 히스토그램을 구하면 가장 오선의 y축 대비 위치를 쉽게 찾을 수 있음
    5% 지점은 오선의 시작 지점으로 임시 지정하고
    좌,우 측 픽셀값을 검사해가며 시작 위치를 추정
    이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    '''
    def __init__(self,img):
        self.__origin_img = cv2.imread(r'SheetMusics/'+img)
        self.__src = super().GrayScale(self.__origin_img)
        self.__dst = self.__src.copy()
        self.hist= []
        self.wpos = []
        self.__values = {}
        self.__h, self.__w = self.__src.shape
        self.__find_hist()
        self.__findFiveLine()
    
    def GetImg(self):
        return self.__src,self.__dst
    
    #  모든 이미지 보기
    @dispatch()
    def show(self):
        cv2.imshow('dst',self.__src)
        cv2.imshow('img',self.__dst)
    
    # 결과 이미지 보기
    @dispatch(str)
    def show(self, name):
        cv2.imshow(name,self.__dst)

    # 악보의 수평 히스토그램을 구하고 그 값중 이미지 너비의 70%이상의
    # 값을 오선으로 판단하고 데이터로 포함
    def __find_hist(self):
        for i in range(1,self.__h-1):
            value = 0
            for j in range(1,self.__w-1):
                if self.__src[i,j] <= 250:
                    value += 1
            if value >= (self.__w/100)*70 :
                self.hist.append(i)
                self.__values[i] = value

    # 악보의 수평 히스토그램을 기반으로 오선의 시작 위치(x축) 추정    
    def __findFiveLine(self):
        s = (self.__w//100)*5

        for h in self.hist:
            if self.__src[h,s] < 240:
                p = 0
                while self.__src[h,s-p] < 240:
                    p += 1
                self.wpos.append(s-p)
            else:
                p = 0
                while self.__src[h,(s+p)+100] >= 240:
                    p += 1
                self.wpos.append(s+p)

    # 검출된 검정 픽셀 선 삭제 
    def delete_line(self,whs):
        for (x,y) in whs:
            for i in range(x,self.__w):
                if self.__dst[y-1,i] >= 180:
                    self.__dst[y,i] = 255
        
        self.__Morph(whs)
                     
    # 모폴로지 사용으로 오선 제거중 사라진 부분 복구
    def __Morph(self, whs):
        morph_img = self.__src.copy()
        morph_img = super().binary(morph_img)
        kernal_v = np.ones((3,3), np.uint8)
        morph_img = cv2.morphologyEx(morph_img,cv2.MORPH_CLOSE, kernal_v)

        for (_,y) in whs:
            for i in range(self.__w):
                if self.__dst[y,i] > morph_img[y,i]:
                    self.__dst[y,i] = morph_img[y,i]

    # 제거된 부분 비율 구하는 부분
    def find_degree(self, whs):
        o_img = self.__src.copy()
        d_img = super().binary(self.__dst.copy())
        avrs = []
        avr = 0
        for (x,y) in whs:
            value = 0
            total = self.__values[y]
            for i in range(x,total):
                if d_img[y,i] < 127:
                    value += 1
            avr = value/total*100
            avrs.append(avr)
        print(f'Total Staff-line Pixel : {total}, Minimum Remained Ratio : {min(avrs):.3f}%')
    