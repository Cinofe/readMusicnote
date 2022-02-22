from multipledispatch import dispatch
import cv2, os, numpy as np

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
        d_img = self.__binary(self.__img.copy())
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
        cv2.imshow('find contour',src)

    # 오선 사이 사이의 가사 또는 잡음 제거
    '''
    2. 오선이 있는 이미지를 이진화 시킨 후 모폴로지 침식으로 어두운 영역을
    한 차례 확장 시킨 후 컨투어로 외관선을 탐지 하면 오선 포함 악보 부분과 
    가사 및 잡음 부분이 분리될꺼같고, 이렇게 분리 되었을 때 히스토그램으로
    찾은 오선의 y축과 겹치지 않는 컨투어는 모두 지우면 그 공간이 지워지지 않을까?
    '''
    '''
        오선 부분 사각형만 체크 확인
        이제 오선부분 사각형 외의 부분 모두 255로 변경 시켜줘야함
        문제 발생 : 오선이 선으로 연결되어 있으면 제대로 제거가 불가능
        그래서 수직 성분을 제거하자니 오선영역 바깥 음표는 영역 내부에 안들어옴.
        
        해결책 : 
        1. 영역의 y축 길이를 위, 아래로 10%정도 씩 늘려보자.
         -> 악보마다 y축 길이가 달라서 보편성이 없음.
        2. 악절의 영역을 먼저 구한 뒤 악보 전체의 특정 영역을 구하고,
        거기서 악절의 영역에 겹치면서 y축 값이 악절의 y축 값보다 크면
        악절 영역을 확장. -> 성공 -> 다른 악보에서 오류 발견 3, 6, 8 악보
        3. 오선의 제거한 이미지에서 윤곽선(contour) 검출 후 악절 영역과 겹치는
        부위를 방법 2 와 같이 진행. (2와 다른점 : 2는 모폴로지로 변환 시킨 이미지
        에서 윤관석(contour)영역을 모두 찾고 해당 영역들 중 악절의 영역과 이외의 영역
        중 겹치는 부위를 비교 분석으로 악절의 영역을 확장, 3은 2에서 최초 악절의 영역만
        찾고 확장시켜야 할 영역을 오선제거된 이미지의 윤곽선(contour) 영역에서 찾아서 
        확장) -> 성공 -> 모든 악보에서 악절 영역만 추출 성공.
        '''
    def delete_noise(self):
        # 기본적인 악절 추적위한 이미지
        src = self.__binary(self.__dst)
        # 악절의 범위 확장을 위한 음표 추적위한 이미지
        src2 = self.__binary(self.__img)
        # 측정된 악절만 이력할 이미지
        src3 = np.full((self.__h,self.__w),255,np.uint8)
        # 악절의 범위를 명확하게 파악하기 위해 형태학(모폴로지)연산 이용
        src = cv2.erode(src, np.ones((3,3), np.uint8),anchor=(-1,-1),iterations=1)
        src = cv2.dilate(src, np.ones((1,5), np.uint8), anchor=(-1,-1),iterations=1)
        # 명확한 음계를 파악하기 위해 형태학(모폴로지)연산 이용
        src2 = cv2.erode(src2, np.ones((2,2),np.uint8),anchor=(-1,-1),iterations=1)
        # 악절이 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        FLlocs = []
        # 기호가 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        SymbolLocs = []
        # 영역의 윤곽을 찾는 연산
        Fcontours, _ = cv2.findContours(src,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        Scontours, _ = cv2.findContours(src2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 악절 영역 찾기
        for contour in Fcontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0 :
                continue
            if h >= 30 and w >= self.__w//2:
                FLlocs.append((x,y,w,h))
                # cv2.rectangle(src,(x,y,w,h),(0,0,0),1)
        
        # 악절 이외의 기호 영역 찾기
        for contour in Scontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0:
                continue
            if w < 10 or h < 20 or h > 70 or (w > 30 and h < 30):
                continue
            SymbolLocs.append((x,y,w,h))
            cv2.rectangle(src2,(x,y),(x+w,y+h),(0,0,0),1)

        # 오선 확장 조건
        # 1. y 조건 - 기호 영역의 y축이 오선의 y축보다 작으면서 기호영역의 y+h가 오선 영역의 y보다 크면 오선 영역의 y를 기호 영역의 y까지 확장.
        # 2. h 조건 - 기호 영역의 y+h가 오선의 y+h보다 크면서 기호 영역의 y가 오선영역의 y+h보다 작으면 오선 영역의 y+h를 기호 영역 y+h까지 확장.
        for _,sy,_,sh in SymbolLocs:
            for j,floc in enumerate(FLlocs):
                fx,fy,fw,fh = floc

                if sy <= fy and (sy + sh) >= fy:
                    fh += fy - sy
                    fy = sy
                
                if (sy + sh) >= (fy + fh) and sy <= (fy + fh):
                    fh += (sy + sh) - (fy + fh)
                
                if not floc is (fx,fy,fw,fh):
                    FLlocs[j] = (fx, fy, fw, fh)       
        # 확장된 악절 영역을 새로운 이미지에 그려주는 작업
        for flloc in FLlocs:
            x,y,w,h = flloc
            src3[y:y+h,x:x+w] = self.__dst[y:y+h,x:x+w].copy()
            cv2.rectangle(self.__dst,flloc,(0,0,0),1)
        """
        제목, 가사들은 제거 완료, 자잘한 노이즈 제거 방법 필요
        노이즈 제거 할때 오선 내부의 음표도 함꼐 지워지는 부분 수정 필요.
        1. 오선 제거된 영상에서 윤관석 검출을 통해 제거해보기
        -> 안됨. 오선과 함께 온음표의 위와 아래가 같이 잘려있어서 노이즈로 처리해버림.
        """
        src4 = src3.copy()
        src4 = self.__binary(src4)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
        src4 = cv2.erode(src4, np.ones((1,1),np.uint8), anchor=(-1,-1),iterations=1)
        # cv2.imshow('test2',src4)
        contours, _ = cv2.findContours(src4, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0:
                continue
            if (w < 13 and h < 11) and (w > 5 or h > 5):
                src3[y:y+h, x:x+w] = np.full((h,w),255,np.uint8)
                cv2.rectangle(src4, (x,y,w,h),(0,0,0),1)
        
        cv2.rectangle(src4,(0,0,13,11),(0,0,0),1)
        cv2.rectangle(src4,(0,12,5,5),(0,0,0),1)
        cv2.imshow('delete noise',self.__dst)
        cv2.imshow('delete five',self.__img)
        cv2.imshow('src',src3)

def allimg():
    imgs = os.listdir(r'SheetMusics')
    for img in imgs:
        # t_start = t.time()
        DFL = Del_FiveLine(img)
        whpos = list(zip(DFL.wpos,DFL.hist))
        DFL.delete_line(whpos)
        # t_end = t.time()
        # print(f"img : {img}, img size(w,h) : {DFL.get_shape()}, process time : {t_end - t_start:.3f}sec")
        DFL.show(img)
        # DFL.find_degree(whpos)
        DFL.delete_noise()
        cv2.waitKey()
    
def oneimg():
    imgs = os.listdir(r'SheetMusics')
    DFL = Del_FiveLine(imgs[5])
    whpos = list(zip(DFL.wpos,DFL.hist))
    DFL.delete_line(whpos)
    # DFL.show()
    # DFL.find_degree(whpos)
    # DFL.find_Contours()
    DFL.delete_noise()

    cv2.waitKey()


def main():
    oneimg()
    # allimg()
    
if __name__ == '__main__':
    main()