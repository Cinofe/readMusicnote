from multipledispatch import dispatch
import matplotlib.pyplot as plt
import cv2, os, numpy as np, time as t

class Del_FiveLine:
    '''
    대부분의 악보들의 오선 시작 위치는 악보의 너비 약 5% 지점에서 시작
    악보의 수직 히스토그램을 구하면 가장 오선의 y축 대비 위치를 쉽게 찾을 수 있음
    5% 지점은 오선의 시작 지점으로 임시 지정하고
    좌,우 측 픽셀값을 검사해가며 시작 위치를 추정
    이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    '''
    def __init__(self,img):
        self.__origin_img = cv2.imread(r'musicnotes/'+img)
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
        cv2.imshow('dst',self.__dst)
        cv2.imshow('img',self.__img)
    
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

    # 제거된 부분 비율 구하는 부분
    def find_degree(self, whs):
        o_img = self.__dst.copy()
        d_img = self.__img.copy()
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
            print(f'Total Pixel : {total}, Remained Pixel : {value}, Remained Ratio : {avr:.3f}%')
        print(f'Total Average : {sum(avrs)/len(avrs):.3f}%')
    
    # 음표 및 여러 객체 외각선 탐지
    def find_Contours(self):
        '''
        1. 음표 인식 방법 모폴로지 침식, 팽창을 이용해서 음표 또는 빔 부분
        좌표를 기억해 놨다가, 컨투어 추출 할 때, 해당 좌표를 포함하고 있는
        부분만 사각형으로 표시 <- 문제점 : 2분 음표 추출 불가
        '''

        src = self.__binary(self.__dst)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
        src = cv2.dilate(src, kernel, anchor=(-1,-1),iterations=2)
        # src = cv2.erode(src, kernel, anchor=(-1,-1),iterations=1)
        
        # contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # i = 1
        # for contour in contours:
        #     x,y,w,h = cv2.boundingRect(contour)
        #     if w < 9 or h < 9:
        #         continue
        #     cv2.rectangle(src,(x,y,w,h),(0,0,0),1)
        #     cv2.putText(src,str(i),(x,y),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),1)
        #     i += 1
        cv2.imshow('origin',self.__dst)
        cv2.imshow('test',src)

    # 오선 사이 사이의 가사 또는 잡음 제거
    '''
    2. 오선이 있는 이미지를 이진화 시킨 후 모폴로지 침식으로 어두운 영역을
    한 차례 확장 시킨 후 컨투어로 외관선을 탐지 하면 오선 포함 악보 부분과 
    가사 및 잡음 부분이 분리될꺼같고, 이렇게 분리 되었을 때 히스토그램으로
    찾은 오선의 y축과 겹치지 않는 컨투어는 모두 지우면 그 공간이 지워지지 않을까?
    '''
    def delete_noise(self, whs):
        src = self.__binary(self.__dst)
        kernel = np.ones((3,3), np.uint8)
        src = cv2.erode(src, kernel,anchor=(-1,-1),iterations=1)
        locs = []
        contours, _ = cv2.findContours(src,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        _, ys = list(zip(*whs))
        print(ys)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            for _y in ys:
                if _y > y and _y < y+h:
                    if h < 20 or w < self.__w//2:
                        continue
                    locs.append((x,y,w,h))
                    cv2.rectangle(self.__img,(x,y,w,h),(0,0,0),1)
        '''
        오선 부분 사각형만 체크 확인
        이제 오선부분 사각형 외의 부분 모두 255로 변경 시켜줘야함
        '''
        cv2.imshow('test',self.__img)

def main():

    imgs = os.listdir(r'musicnotes')
    for img in imgs:
        t_start = t.time()
        DFL = Del_FiveLine(img)
        whpos = list(zip(DFL.wpos,DFL.hist))
        DFL.delete_line(whpos)
        t_end = t.time()
        print(f"img : {img}, img size(w,h) : {DFL.get_shape()}, process time : {t_end - t_start:.3f}sec")
        # DFL.show(img)
    #     DFL.find_degree(whpos)
    #     DFL.delete_noise(whpos)
    #     cv2.waitKey()

    # DFL = Del_FiveLine(imgs[3])
    # whpos = list(zip(DFL.wpos,DFL.hist))
    # DFL.delete_line(whpos)
    # DFL.show()
    # DFL.find_degree(whpos)
    # DFL.find_Contours()
    # DFL.delete_noise(whpos)

    cv2.waitKey()
    
if __name__ == '__main__':
    main()