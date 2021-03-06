from multipledispatch import dispatch
import numpy as np, cv2

class Del_FiveLine:
    '''
    대부분의 악보들의 오선 시작 위치는 악보의 너비 약 5% 지점에서 시작
    악보의 수직 히스토그램을 구하면 가장 오선의 y축 대비 위치를 쉽게 찾을 수 있음
    5% 지점은 오선의 시작 지점으로 임시 지정하고
    좌,우 측 픽셀값을 검사해가며 시작 위치를 추정
    이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    '''
    def __init__(self,img):
        self.__origin_img = cv2.imread(r'SheetMusics/'+img)
        self.__dst = []
        self.__img = []
        self.hist= []
        self.wpos = []
        self.__values = {}
        self.__GrayScale()
        self.__h, self.__w = self.__dst.shape
        self.__find_hist()
        self.__findFiveLine()
    
    def get_shape(self):
        return (self.__w,self.__h)
    
    def __str__(self):
        return f'hist : {self.hist} hist length : {len(self.hist)}\nwpos : {self.wpos} wpos length : {len(self.wpos)}'
    
    # 이미지를 컬러 영상에서 -> 흑백 영상으로 변환
    def __GrayScale(self):
        self.__dst = cv2.cvtColor(self.__origin_img,cv2.COLOR_BGR2GRAY)
        self.__img = self.__dst.copy()
    
    #  모든 이미지 보기
    @dispatch()
    def show(self):
        cv2.imshow('origin',self.__dst)
        cv2.imshow('delete',self.__img)
    
    # 결과 이미지 보기
    @dispatch(str)
    def show(self, name):
        cv2.imshow(name,self.__img)

    # 악보의 수평 히스토그램을 구하고 그 값중 이미지 너비의 70%이상의
    # 값을 오선으로 판단하고 데이터로 포함
    def __find_hist(self):
        # hist = []
        # ys = []
        for i in range(1,self.__h-1):
            value = 0
            for j in range(1,self.__w-1):
                if self.__dst[i,j] <= 250:
                    value += 1
            if value >= (self.__w/100)*70 :
                self.hist.append(i)
                self.__values[i] = value
        #     hist.append(value)
        #     ys.append(i)
        
        # plt.barh(ys,hist)
        # plt.show()

    # 악보 위나 아래의 필요없는 부분 삭제
    def __delete_Name(self):
        miny = min(self.hist)
        maxy = max(self.hist)
        avr = (self.__h//100)*5
        self.__img = self.__img[miny-avr:maxy+avr,0:self.__w].copy()
        self.__dst = self.__dst[miny-avr:maxy+avr,0:self.__w].copy()

    # 악보의 수평 히스토그램을 기반으로 오선의 시작 위치(x축) 추정    
    def __findFiveLine(self):
        s = (self.__w//100)*5

        for h in self.hist:
            if self.__dst[h,s] < 240:
                p = 0
                while self.__dst[h,s-p] < 240:
                    p += 1
                self.wpos.append(s-p)
            else:
                p = 0
                while self.__dst[h,(s+p)+100] >= 240:
                    p += 1
                self.wpos.append(s+p)

    # 검출된 검정 픽셀 선 삭제 
    def delete_line(self,whpos):
        for (x,y) in whpos:
            for i in range(x,self.__w):
                if self.__img[y-1,i] >= 180:
                    self.__img[y,i] = 255
        self.__Morph(whpos)
        # self.__delete_Name()

    # 이미지 이진화
    def __binary(self, img):
        _, new_img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        return new_img
                     
    # 모폴로지 사용으로 오선 제거중 사라진 부분 복구
    def __Morph(self, whpos):
        morph_img = self.__dst.copy()
        morph_img = self.__binary(morph_img)
        kernal_v = np.ones((3,3), np.uint8)
        morph_img = cv2.morphologyEx(morph_img,cv2.MORPH_CLOSE, kernal_v)

        for (_,y) in whpos:
            for i in range(self.__w):
                if self.__img[y,i] > morph_img[y,i]:
                    self.__img[y,i] = morph_img[y,i]