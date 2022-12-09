import cv2, numpy as np, matplotlib.pyplot as plt 

class antiAliasing:
    def __init__(self):
        self.__src = []
        self.__dst = []
        ## 탐색할 방향 지정
        self.__Os = {'O1':[(0,-2),(0,-1),(0,0),(0,1),(0,2)],
                    'O2':[(2,-2),(1,-1),(0,0),(-1,1),(-2,2)],
                    'O3':[(2,0),(1,0),(0,0),(-1,0),(-2,0)],
                    'O4':[(2,2),(1,1),(0,0),(-1,-1),(-2,-2)]}
    
    def Bin(self,img):
        _,th = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
        return th

    def detectLaplace(self, img):
        return self.Bin((255)-cv2.Laplacian(img, -1))

    def antiAliasing(self, img):
        self.__src = self.Bin(img)
        self.__dst = self.__src.copy()
        Lap_src = self.detectLaplace(self.__src)
        ## 경계선 좌표 추출
        pos = []
        for y, i in enumerate(Lap_src):
            pos.extend([(x,y)for x,v in enumerate(i) if v == 0])
        ## 지정 방향 탐색 및 조건에 따른 Anti Aliasing
        for x, y in pos:
            O_value = {'O1':[],'O2':[],'O3':[],'O4':[]}
            for O in self.__Os:
                cnts = []
                for Ox, Oy in self.__Os.get(O):
                    try:
                        O_value.get(O).append(1 if self.__src[y+Oy][x+Ox] == 0 else 0)
                    except Exception as e:
                        continue
                cnts.append(O_value.get(O).count(1))
                ## 탐진된 검정 픽셀 수가 1이면 해당 픽셀을 흰색으로
                if 1 in cnts:
                    self.__dst[y][x] = 255
                    break
                ## 탐지된 검정 픽셀 수가 5이면 해당 없음
                elif 5 in cnts:
                    break
                ## 그 이외의 경우 탐지된 검정 픽셀 값의 순서가 흑,백,흑 일 경우 백의 자리를 흑으로 변환
                else:
                    for cnt in cnts:
                        if cnt<3:
                            v = O_value.get(O)
                            for i in range(len(v)-2):
                                if v[i] == 1 and v[i+1] == 0 and v[i+2] == 1:
                                    ox,oy = self.__Os.get(O)[i+1]
                                    self.__dst[y+oy][x+ox] = 0
                                    break
                            break
        return self.__dst

    def show(self):
        merged = np.hstack([self.__src, self.__dst])
        plt.figure(figsize=(17,6))
        plt.imshow(merged)
        plt.show()


if __name__ == "__main__":
    # 원본
    src = cv2.imread(r'Find_Symbols/73.jpg', 0)

    ali = antiAliasing()
    src = ali.antiAliasing(src)
    ali.show()
