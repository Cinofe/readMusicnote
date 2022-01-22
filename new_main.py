from multipledispatch import dispatch
import cv2, os, time as t, numpy as np, matplotlib.pyplot as plt

class Del_FiveLine:
    '''
    대부분의 악보들의 오선 시작 위치는 악보의 너비 약 5% 지점에서 시작
    악보의 수직 히스토그램을 구하면 가장 오선의 y축 대비 위치를 쉽게 찾을 수 있음
    5% 지점은 오선의 시작 지점으로 임시 지정하고
    좌,우 측 픽셀값을 검사해가며 시작 위치를 추정
    이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    '''
    def __init__(self,img) -> None:
        self.__origin_img = cv2.imread(r'musicnotes/'+img)
        self.__dst = []
        self.__img = []
        self.hist= []
        self.wpos = []
        self.__GrayScale()
        self.__h, self.__w = self.__dst.shape
        self.__find_hist()
        self.__findFiveLine()
    
    def __str__(self) -> str:
        return f'hist : {self.hist} hist length : {len(self.hist)}\nwpos : {self.wpos} wpos length : {len(self.wpos)}'
    
    # 이미지를 컬러 영상에서 -> 흑백 영상으로 변환
    def __GrayScale(self) -> None:
        self.__dst = cv2.cvtColor(self.__origin_img,cv2.COLOR_BGR2GRAY)
        self.__img = self.__dst.copy()
    
    #  모든 이미지 보기
    @dispatch()
    def show(self) -> None:
        cv2.imshow('dst',self.__dst)
        cv2.imshow('img',self.__img)
    
    # 결과 이미지 보기
    @dispatch(str)
    def show(self, name):
        cv2.imshow(name,self.__img)

    # 악보의 수평 히스토그램을 구하고 그 값중 이미지 너비의 70%이상의
    # 값을 오선으로 판단하고 데이터로 포함
    def __find_hist(self) -> None:
        for i in range(1,self.__h-1):
            value = 0
            for j in range(1,self.__w-1):
                if self.__dst[i,j] == self.__dst[i,j-1]:
                    weight = 1.2
                else:
                    weight = 1
                if self.__dst[i,j] < 245:
                    value += weight
            if value >= (self.__w/100)*70 :
                self.hist.append(i)
        
    
    # 악보의 수평 히스토그램을 기반으로 오선의 시작 위치(x축) 추정    
    def __findFiveLine(self) -> None:
        s = (self.__w//100)*5

        for h in self.hist:
            if self.__dst[h,s] < 240:
                p = 0
                while self.__dst[h,s-p] < 240:
                    p += 1
                self.wpos.append((s-p)+1)
            else:
                p = 0
                while self.__dst[h,(s+p)] >= 240:
                    p += 1
                self.wpos.append((s+p)-1)

    # 검출된 검정 픽셀 선 삭제 
    def delete_line(self,whpos) -> None:
        for (x,y) in whpos:
            for i in range(x,self.__w):
                if self.__img[y-1,i] >= 180:
                    self.__img[y,i] = 255
        self.__Morph(whpos)

    # 이미지 이진화
    def binary(self) -> None:
        # _, self_dst = cv2.threshold(self.__dst,127,255,cv2.THRESH_OTSU)
        for i in range(self.__img.shape[0]):
            for j in range(self.__img.shape[1]):
                if self.__img[i,j] > 130:
                    self.__img[i,j] = 255
                else:
                    self.__img[i,j] = 0
                     
    # 모폴로지 사용으로 오선 제거중 사라진 부분 복구
    def __Morph(self, whpos) -> None:
        morph_img = self.__dst.copy()
        _, morph_img = cv2.threshold(morph_img,127,255,cv2.THRESH_OTSU)
        kernal_v = np.ones((3,3), np.uint8)
        morph_img = cv2.morphologyEx(morph_img,cv2.MORPH_CLOSE, kernal_v)
        cv2.imshow('test3',morph_img)

        for (_,y) in whpos:
            for i in range(self.__w):
                if self.__img[y,i] > morph_img[y,i]:
                    self.__img[y,i] = morph_img[y,i]

def main():

    imgs = os.listdir(r'musicnotes')
    # for img in imgs:
    #     DFL = Del_FiveLine(img)
    #     whpos = list(zip(DFL.wpos,DFL.hist))
    #     DFL.delete_line(whpos)
    #     DFL.show(img)

    DFL = Del_FiveLine(imgs[0])
    whpos = list(zip(DFL.wpos,DFL.hist))
    DFL.delete_line(whpos)
    DFL.show()

    cv2.waitKey()
    
if __name__ == '__main__':
    main()