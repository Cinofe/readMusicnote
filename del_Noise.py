from parent import parent
import cv2, numpy as np, os

class Del_Noise(parent):
    def __init__(self, src, dst, hist):
        self.__src = src
        self.__dst = dst
        self.__h, self.__w = self.__src.shape
        self.__img = []
        self.hist = hist
        # 악절이 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        self.FLlocs = []
        # 기호가 표현된 윤곽 사각형의 좌표가 들어갈 리스트
        self.SymbolLocs = []

    def delete_noise(self):
        # 기본적인 악절 추적위한 이미지
        dst = super().binary(self.__src)
        # 악절의 범위 확장을 위한 음표 추적위한 이미지
        dst2 = super().binary(self.__dst)
        # cv2.imshow('test',self.__dst)
        # 측정된 악절만 이력할 이미지
        dst3 = np.full((self.__h,self.__w),255,np.uint8)
        dst4 = np.full((self.__h,self.__w),255,np.uint8)
        # 악절의 범위를 명확하게 파악하기 위해 형태학(모폴로지)연산 이용
        dst = cv2.erode(dst, np.ones((3,3), np.uint8),anchor=(-1,-1),iterations=1)
        dst = cv2.dilate(dst, np.ones((1,5), np.uint8), anchor=(-1,-1),iterations=1)
        # 명확한 음계를 파악하기 위해 형태학(모폴로지)연산 이용
        dst2 = cv2.erode(dst2, np.ones((1,2),np.uint8),anchor=(-1,-1),iterations=1)

        # 영역의 윤곽을 찾는 연산
        Fcontours, _ = cv2.findContours(dst,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        Scontours, _ = cv2.findContours(dst2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 악절 영역 찾기
        for contour in Fcontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0 :
                continue
            if h >= 30 and w >= self.__w//2:
                self.FLlocs.append((x,y,w,h))
                
        
        # 악절 이외의 기호 영역 찾기
        for contour in Scontours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 and y == 0:
                continue
            if (w < 10 or w > 70) or (h < 18 or h > 70) or (w > 25 and h < 25) or (w < 20 and h < 20):
                continue
            self.SymbolLocs.append((x,y,w,h))
        
        # 오선 확장 조건
        # 1. y 조건 - 기호 영역의 y축이 오선의 y축보다 작으면서 기호영역의 y+h가 오선 영역의 y보다 크면 오선 영역의 y를 기호 영역의 y까지 확장.
        # 2. h 조건 - 기호 영역의 y+h가 오선의 y+h보다 크면서 기호 영역의 y가 오선영역의 y+h보다 작으면 오선 영역의 y+h를 기호 영역 y+h까지 확장.
        for _,sy,_,sh in self.SymbolLocs:
            for j,floc in enumerate(self.FLlocs):
                fx,fy,fw,fh = floc

                if sy <= fy and (sy + sh) >= fy:
                    fh += fy - sy
                    fy = sy
                
                if (sy + sh) >= (fy + fh) and sy <= (fy + fh):
                    fh += (sy + sh) - (fy + fh)
                
                if not floc is (fx,fy,fw,fh):
                    self.FLlocs[j] = (fx, fy, fw, fh)
        
        # 확장된 악절 영역을 새로운 이미지에 그려주는 작업
        for flloc in self.FLlocs:
            x,y,w,h = flloc
            dst3[y:y+h,x:x+w] = self.__src[y:y+h,x:x+w].copy()
            dst4[y:y+h,x:x+w] = self.__dst[y:y+h,x:x+w].copy()

        self.__img = dst3.copy()
        self.__dst = dst4.copy()

    # 음표 및 여러 객체 찾아서 저장
    def find_Contours(self):
        src = super().binary(self.__dst)
        dst = super().binary(self.__dst)

        contours, _ = cv2.findContours(src,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        dir = os.listdir(r'Find_Symbols/')
        # i = len(dir)
        i = 1

        if len(dir) != 0:
            for d in dir:
                os.remove(r'Find_Symbols/'+d)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if x == 0 or y == 0:
                continue

            if w>7 or h>7:
                # 효율적인 인식을 위해 여백 공간 확보
                new_img = dst[y-3:y+h+3,x-3:x+w+3].copy()

                ## 튜닝의 끝은 순정
                cv2.imwrite(r'Find_Symbols/'+str(i)+".jpg",new_img)
                i += 1


    def findData(self):

        src = self.__dst.copy()
        src = super().binary(src)

        ## 오선이 포함된 음표 추출음 위한 함수
        ## 필요 정보
        # 악절 영역의 y와 h 좌표 값
        # 음표 영역의 좌표
        # 빔 음표의 구분을 위한 음표 헤드의 영역 좌표
        # 3가지 정보를 조합해 자르는 크기 정하기

        ### 악절 영역의 y와 h 좌표 값
        FLlocs = self.FLlocs

        ### 음표 영역의 각 좌표
        SyLocs = self.SymbolLocs

        ### 음표 헤드 영역 좌표
        SymHeads = []

        ### 추출된 데이터 셋 모음
        dataSet = []

        ### 빔 음표 구분을 위한 음표 헤드 영역 추출
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1))
        src = cv2.dilate(src,kernel,anchor=(-1,-1),iterations=1)
        src = cv2.erode(src,kernel,anchor=(-1,-1),iterations=1)

        contours, _ = cv2.findContours(src, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

        dst = cv2.cvtColor(self.__src.copy(),cv2.COLOR_GRAY2BGR)
        ## 음표 헤드 영역 찾기
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 and h > 5:
                SymHeads.append((x,y,w,h))

        SymHeads.sort(key=lambda x : (x[1],x[0]))
        FLlocs.sort(key = lambda x : (x[1],x[0]))
        SyLocs.sort(key = lambda x : (x[1],x[0]) )

        ## 음표 분리 하기
        for _, F_y, _, F_h in FLlocs:
            for i, [_, S_y, _, S_h] in enumerate(SyLocs):
                if S_y > F_y and S_y < (F_y + F_h):
                    for j, [sy_x, sy_y, sy_w, _] in enumerate(SymHeads):    
                        if sy_y > S_y and sy_y < (S_y + S_h):
                            symbol = dst[F_y:F_y+F_h,sy_x-7:sy_x+sy_w+7].copy()
                            dataSet.append(symbol)
                            SymHeads.pop(j)
                    SyLocs.pop(i)
        
        ### 추출
        for i, data in enumerate(dataSet):
            data = cv2.resize(data,dsize=None,fx=3,fy=3,interpolation=cv2.INTER_LANCZOS4)
            # cv2.imwrite(rf"Find_Symbols(Staff-Line)/{i}.jpg",data)


if __name__ == "__main__":
    import MainProgram
    MainProgram.main()