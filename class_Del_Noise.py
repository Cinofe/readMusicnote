from parent import parent
import cv2, numpy as np, os
'''
    오류 발생 검토 및 수정 필요
'''
class Del_Noise(parent):
    def __init__(self, src, dst, hist):
        self.__src = src
        self.__dst = dst
        self.__h, self.__w = self.__src.shape
        self.__img = []
        self.hist = hist

    ## 여기 부터는 Delete Five Line Class가 아닌 
    ## Delete Noise Class 를 만들어서 옮기기
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
        print("dst :",end= " ")
        dst = super().binary(self.__src)
        # 악절의 범위 확장을 위한 음표 추적위한 이미지
        print("dst2 :",end= " ")
        dst2 = super().binary(self.__dst)
        cv2.imshow('test',self.__dst)
        # 측정된 악절만 이력할 이미지
        dst3 = np.full((self.__h,self.__w),255,np.uint8)
        dst4 = np.full((self.__h,self.__w),255,np.uint8)
        # 악절의 범위를 명확하게 파악하기 위해 형태학(모폴로지)연산 이용
        dst = cv2.erode(dst, np.ones((3,3), np.uint8),anchor=(-1,-1),iterations=1)
        dst = cv2.dilate(dst, np.ones((1,5), np.uint8), anchor=(-1,-1),iterations=1)
        # 명확한 음계를 파악하기 위해 형태학(모폴로지)연산 이용
        # dst2 = cv2.erode(dst2, np.ones((1,2),np.uint8),anchor=(-1,-1),iterations=1)
        # 악절이 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        FLlocs = []
        # 기호가 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        SymbolLocs = []
        # 영역의 윤곽을 찾는 연산
        Fcontours, _ = cv2.findContours(dst,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        Scontours, _ = cv2.findContours(dst2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 악절 영역 찾기
        for contour in Fcontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0 :
                continue
            if h >= 30 and w >= self.__w//2:
                FLlocs.append((x,y,w,h))
                cv2.rectangle(dst,(x,y),(x+w,y+h),(0,0,0),1)
        
        # 악절 이외의 기호 영역 찾기
        for contour in Scontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0:
                continue
            if (w < 10 or w > 70) or (h < 18 or h > 70) or (w > 25 and h < 25) or (w < 20 and h < 20):
                continue
            SymbolLocs.append((x,y,w,h))
            cv2.rectangle(dst2,(x,y),(x+w,y+h),(0,0,0),1)
        # cv2.imshow("dst",dst)
        # cv2.imshow("dst2",dst2)
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
            dst3[y:y+h,x:x+w] = self.__src[y:y+h,x:x+w].copy()
            dst4[y:y+h,x:x+w] = self.__dst[y:y+h,x:x+w].copy()
            cv2.rectangle(self.__src,flloc,(0,0,0),1)
        # cv2.imshow('src',self.__src)
        # cv2.waitKey()
        self.__img = dst3.copy()
        self.__dst = dst4.copy()
        """
        제목, 가사들은 제거 완료, 자잘한 노이즈 제거 방법 필요
        노이즈 제거 할때 오선 내부의 음표도 함꼐 지워지는 부분 수정 필요.
        1. 오선 제거된 영상에서 윤관석 검출을 통해 제거해보기
        -> 안됨. 오선과 함께 온음표의 위와 아래가 같이 잘려있어서 노이즈로 처리해버림.
        
        일단 세세한 잡음제거는 일시 정지
         - 세세한 잡음은 무시하고 노래 제목, 작사, 작곡, 가사 등의 큰 잡음만 제거 후
           기호를 찾고 잘라서 분류하는것 부터 진행.
        """
        self.__delete_Name()

    # 악보 위나 아래의 필요없는 부분 삭제
    def __delete_Name(self):
        miny = min(self.hist)
        maxy = max(self.hist)
        avr = (self.__h//100)*5
        self.__dst = self.__dst[miny-avr:maxy+avr,0:self.__w].copy()
        self.__src = self.__src[miny-avr:maxy+avr,0:self.__w].copy()
        self.__img = self.__img[miny-avr:maxy+avr,0:self.__w].copy()

# 음표 및 여러 객체 찾아서 저장
    def find_Contours(self):
        src = super().binary(self.__dst)
        dst = super().binary(self.__dst)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
        src = cv2.erode(src,kernel,anchor=(-1,-1),iterations=1)

        contours, _ = cv2.findContours(src,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        """
        윤곽선만 잡아서 이미지를 잘라내려고 했는데 언떤결에 세세한 잡음을 어느정도 제거하는데 성공함
        -> 완전히 제거 하려고 하지 말고, 각 이미지를 잘라 저장하고, 저장한 이미지에서 세선화를 거쳐
        패턴을 찾고 분류하려는 패턴 이외의 기호들은 잡음으로 처리
        """
        i = 0
        dir = os.listdir(r'Find_Symbols/')

        if len(dir) != 0:
            for d in dir:
                os.remove(r'Find_Symbols/'+d)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 or y == 0:
                continue

            if w>7 or h>7:
                new_img = dst[y:y+h,x:x+w].copy()
                cv2.imwrite(r'Find_Symbols/'+str(i)+".jpg",new_img)
                i += 1