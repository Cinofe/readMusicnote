import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 음표 분류


class Classification:
    def __init__(self):
        self.__sift = cv2.xfeatures2d.SIFT_create()
        self.__sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 파라미터 template 들어 있는 폴더명, compare 이미지 들어 있는 폴더명
    # 이미지 높이, 너비 반올림 확장

    def delete_Noise(self, path):
        imgs = os.listdir(path)
        print(imgs)
        for img in imgs:
            src = cv2.imread(path+img)
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

            h, w = src.shape
            nw = [255 for _ in range(w)]
            nh = [255 for _ in range(h)]
            src[0][0:w] = nw
            src[h-1][0:w] = nw
            src = np.array(src).T
            src[0][0:h] = nh
            src[w-1][0:h] = nh
            src = np.array(src).T

            cv2.imwrite(path+img, src)
            plt.imshow(src, plt.cm.gray)
            plt.show()

    def Standardization(self, t_path, c_path):
        t_imgs = os.listdir(t_path)
        c_imgs = os.listdir(c_path)

        def get_new(p):
            return int(np.ceil(p/10)*10) if p % 10 != 0 else p

        def get_empty(p):
            return [255 for _ in range(p)]

        def new_line(src, n, n2, p):
            for _ in range(n):
                src.insert(0, get_empty(p))
            for _ in range(n2):
                src.append(get_empty(p))
            return src

        def loop(imgs, path):
            for img in imgs:
                src = cv2.cvtColor(cv2.imread(path+img), cv2.COLOR_RGB2GRAY)
                h, w = src.shape

                nh = get_new(h)
                nw = get_new(w)

                l = int((nw-w)//2)
                r = (nw-w)-l
                u = int((nh-h)//2)
                d = (nh-h)-u

                src = list(src)
                src = new_line(src, u, d, w)
                src = list(np.array(src).T)
                src = new_line(src, l, r, h+u+d)
                src = np.array(src).T

                cv2.imwrite(path+img, src)

        loop(t_imgs, t_path)
        loop(c_imgs, c_path)

    def One_Resize(self, img, Scale, interpol):
        src = img.copy()
        h, w = src.shape
        src = cv2.resize(src, (w*Scale, h*Scale), interpolation=interpol)
        _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
        return src

    def Repeat_Reszie(self, img, Scale, interpol):
        src = img.copy()
        _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
        h, w = src.shape
        ph = int((Scale-1)*h//10)
        pw = int((Scale-1)*w//10)

        for _ in range(10):
            h, w = src.shape
            src = cv2.resize(src, (w+pw, h+ph), interpolation=interpol)
        _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

        return src

    def Resize(self, imgs, S, INTER):
        t_img, c_img = imgs
        ch, cw = c_img.shape

        t_img = cv2.resize(t_img, (cw, ch), interpolation=cv2.INTER_AREA)

        Ot_img = self.One_Resize(t_img, S, INTER)
        Rt_img = self.Repeat_Reszie(t_img, S, INTER)
        Oc_img = self.One_Resize(c_img, S, INTER)
        Rc_img = self.Repeat_Reszie(c_img, S, INTER)

        self.Similarity([Ot_img, Oc_img], [Rt_img, Rc_img])

    def Similarity(self, Os, Rs):

        def GoodMatch(matches, kpt):
            ts = [*reversed(Ot_img.shape)]
            cs = [*reversed(Oc_img.shape)]
            w_Ratio = ((5*cs[0])/(2*ts[0]))+((5*ts[0])/(2*cs[0]))+10
            h_Ratio = ((5*cs[1])/(2*ts[1]))+((5*ts[1])/(2*cs[1]))+10

            new_matches = []
            for mat in matches:
                kp1 = [*map(int, kpt[0][mat.queryIdx].pt)]
                kp2 = [*map(int, kpt[1][mat.trainIdx].pt)]
                t_Ratio = [kp/ts[i]*100 for i, kp in enumerate(kp1)]
                c_Ratio = [kp/cs[i]*100 for i, kp in enumerate(kp2)]
                # 두 좌표 비율의 차가 +- 두 이미지 대비 비율차 15% 이내 일 경우 좋은 매칭이라 판단
                if abs(t_Ratio[0] - c_Ratio[0]) <= w_Ratio and abs(t_Ratio[1] - c_Ratio[1]) <= h_Ratio:
                    new_matches.append(mat)

            return new_matches


        def Classification(t_img,c_img,sifts,gMatch,f,it):
            # Find Good Match
            cnt = 0
            p = []
            print(f'SIFT Total Matching : {len(sifts)}, Good Matching : {len(gMatch)}')
            img = cv2.drawMatches(t_img, sifts[0][0], c_img, sifts[0][1], gMatch, None, flags=f)

            h, w = t_img.shape
            if w < 150 or h < 200:
                kernal1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                n_img = np.zeros((h, w),np.uint8)
                for i in range(h):
                    for j in range(w):
                        if t_img[i][j] == c_img[i][j]:
                            n_img[i][j] = 255
                        else:
                            n_img[i][j] = 0
                
                n_img = cv2.dilate(n_img,kernal1,iterations=it)
                _, n_img = cv2.threshold(n_img, 0, 255, cv2.THRESH_OTSU)
                conts, hier = cv2.findContours(n_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                n_img = cv2.cvtColor(n_img, cv2.COLOR_GRAY2RGB)

                for cont in conts:
                    x, y, w, h = cv2.boundingRect(cont)
                    if not x and not y:
                        continue
                    if w > 15 and h > 25:
                        # cv2.rectangle(n_img, (x,y), (x+w, y+h), (0,0,255))
                        cnt += 1

            ## 음표 이미지 분류
            if cnt == 0:
                print(t[:2])
                p += os.listdir(r"Classific_Symbols/8b/")+os.listdir(r"Classific_Symbols/8/")+os.listdir(r"Classific_Symbols/4/")+os.listdir(r"Classific_Symbols/2/")
                print(p)
                if c not in p:
                    if t[0] == '8':
                        if t[1] == 'b':
                            cv2.imwrite(r"Classific_Symbols/8b/"+c,c_img)
                        else:
                            cv2.imwrite(r"Classific_Symbols/8/"+c,c_img)
                    elif t[0] == '4':
                        cv2.imwrite(r"Classific_Symbols/4/"+c,c_img)
                    elif t[0] == '2':
                        cv2.imwrite(r"Classific_Symbols/2/"+c,c_img)
            cv2.imshow("img", img)
            # cv2.moveWindow("img", 300, 300)
        

        def getPoint(t_img, c_img):
            kp1, des1 = self.__sift.detectAndCompute(t_img,None)
            kp2, des2 = self.__sift.detectAndCompute(c_img,None)
            if len(kp1) != 0 and len(kp2) != 0 and len(des1) != 0 and len(des2) != 0:
                return [(kp1, kp2), (des1, des2)]
            else:
                return 0

        Ot_img, Oc_img = Os
        Rt_img, Rc_img = Rs

        O_sifts = getPoint(Ot_img, Oc_img)
        if not O_sifts:
            return
        
        R_sifts = getPoint(Rt_img, Rc_img)
        if not R_sifts:
            return

        O_sift_matches = self.__sift_matcher.match(*O_sifts[1])
        R_sift_matches = self.__sift_matcher.match(*R_sifts[1])

        if len(O_sift_matches) > 5 and len(R_sift_matches) > 5:
            G_O_Match = GoodMatch(O_sift_matches, O_sifts[0])
            G_R_Match = GoodMatch(R_sift_matches, R_sifts[0])

            flag = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

            i = 6
            if len(G_O_Match) > len(O_sift_matches)//2:
                Classification(Ot_img, Oc_img, O_sifts, G_O_Match,flag, it=i)

            if len(G_R_Match) > len(R_sift_matches)//2:
                Classification(Rt_img, Rc_img, R_sifts, G_R_Match, flag,  it=i)
                
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()

## 비슷하게 생겨서 잘못 인식된 이미지들 분류
## 위 결과 확인하면서 임계값 조정 하기

if __name__ == "__main__":
    cla = Classification()
    # cla.delete_Noise("template/")
    t_imgs = os.listdir("template/")
    c_imgs = sorted(os.listdir("Find_Symbols/"),key=lambda x:int(x[:-4]))
    # cla.Standardization("template/", "compare/")
    for t in t_imgs:
        t_img = cv2.imread("template/"+t, cv2.IMREAD_GRAYSCALE)
        for c in c_imgs:
            # os.system('cls')
            # print(c)
            c_img = cv2.imread("Find_Symbols/"+c, cv2.IMREAD_GRAYSCALE)
            cla.Resize([t_img, c_img], 9, cv2.INTER_LANCZOS4)
