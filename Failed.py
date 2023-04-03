import cv2, math
import os
import numpy as np
import matplotlib.pyplot as plt

# 음표 분류
class Classification:
    # def __init__(self):
        # self.__sift = cv2.xfeatures2d.SIFT_create()
        # self.__sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 파라미터 template 들어 있는 폴더명, compare 이미지 들어 있는 폴더명
    # 이미지 높이, 너비 반올림 확장
    # 이미지 가장자리 검정 선을 흰선으로 바꿔줌
    def delete_Noise(self, path):
        imgs = os.listdir(path)
        # print(imgs)
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

            contour, hr = cv2.findContours(src,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for cont in contour:
                x, y, w, h = cv2.boundingRect(cont)
                if not x and not y:
                    continue
                if (x-20) > 0 and (y-20) > 0 and (x+w+40) < src.shape[1] and (y+h+40) < src.shape[0] and w >100 and h > 100:
                    # cv2.rectangle(src, (x-20,y-20), (x+w+40, y+h+40), (0,0,255))
                    src = src[y-20:y+h+30,x-20:x+w+30].copy()

            cv2.imwrite(path+img, src)
            # plt.imshow(src, plt.cm.gray)
            # plt.show()
    # 두 비교 이미지 각각 가로 세로 크기를 10의 배수로 변경하기 위한 코드
    def Standardization(self, t_path, c_path):
        t_imgs = os.listdir(t_path)
        c_imgs = os.listdir(c_path)

        def get_new(p):
            ## np.ceil : 주어진 숫자와 같은 정수 또는 주어진 숫자보다 크며 가장 가까운 정수를 반환
            return int(np.ceil(p/10)*10) if p % 10 != 0 else p

        def get_empty(p):
            ## 흰색 선
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
    # 한번에 원하는 배율로 이미지를 확장하기 위한 코드
    def One_Resize(self, img, Scale, interpol):
        src = img.copy()
        h, w = src.shape
        src = cv2.resize(src, (w*Scale, h*Scale), interpolation=interpol)
        # _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
        return src
    
    # 10번에 걸쳐 점차 원하는 배율로 이미지를 확장하는 코드
    def Repeat_Reszie(self, img, Scale, interpol):
        src = img.copy()
        # _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
        h, w = src.shape
        ph = int((Scale-1)*h//10)
        pw = int((Scale-1)*w//10)

        for _ in range(10):
            h, w = src.shape
            src = cv2.resize(src, (w+pw, h+ph), interpolation=interpol)
        # _, src = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

        return src
    ## 길다란 빔 음표의 경우 비교 이미지 크기로 축소시 깨지는 부분이 많아져 분류 정확도가 급속히 낮아짐.
    # 이미지 크기를 변경한 다음 유사도를 비교하기 위한 코드
    def Resize(self, imgs, S, INTER):
        t_img, c_img = imgs
        # ch, cw = c_img.shape

        # t_img = cv2.resize(t_img, (cw, ch), interpolation=cv2.INTER_AREA)

        Ot_img = self.One_Resize(t_img, S, INTER)
        Rt_img = self.Repeat_Reszie(t_img, S, INTER) 
        Oc_img = self.One_Resize(c_img, S, INTER)
        Rc_img = self.Repeat_Reszie(c_img, S, INTER)

        self.Similarity([Ot_img, Oc_img], [Rt_img, Rc_img])

    def Similarity(self, Os, Rs):

        def GoodMatch(matches, kpt):
            img1 = Ot_img.copy()
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
            img2 = Oc_img.copy()
            img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            # 각 이미지의 크기 (w,h) <- img.shape의 출력은 (h,w)이기 떄문에 순서를 바꿔줌
            ts = [*reversed(Ot_img.shape)]
            cs = [*reversed(Oc_img.shape)]

            new_matches = []
            for mat in matches:
                # t_img의 keyPoint 좌표
                kp1 = [*map(int, kpt[0][mat.queryIdx].pt)]
                # c_img의 keyPoint 좌표
                kp2 = [*map(int, kpt[1][mat.trainIdx].pt)]
                # 각 keypoint의 좌표 위치 비율 계산
                t_Ratio = [kp/ts[i]*100 for i, kp in enumerate(kp1)]
                c_Ratio = [kp/cs[i]*100 for i, kp in enumerate(kp2)]
                # 두 좌표 비율의 차가 +- 두 이미지 대비 비율차 18% 이내 일 경우 좋은 매칭이라 판단
                if abs(t_Ratio[0] - c_Ratio[0]) <= 18 and abs(t_Ratio[1] - c_Ratio[1]) <= 18:
                    new_matches.append(mat)

            return new_matches
        ## t_img를 c_img 크기로 맞춘 후 두 이미지 픽셀 비교
        def Classification(t_img,c_img,it):
            # Find Good Match
            cnt = 0

            h,w = c_img.shape
            # 일반적으로 t_img가 c_img 보다 크기 때문에 inter area 적용
            t_img = cv2.resize(t_img,(w,h),interpolation=cv2.INTER_AREA)
            cv2.imshow('test',t_img)
            cv2.waitKey()
            # 모폴로지 연산에 사용될 커널(마스크)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # t_img에 모폴로지 침식 적용시키기
            t_img = cv2.dilate(t_img,k,iterations=4)
            cv2.imshow("test",t_img)
            cv2.waitKey()
            # 두 이미지의 픽셀 값의 변화에 따른 결과 이미지를 담을 새로운 리스트
            n_img = np.zeros((h, w),np.uint8)
            ## 두 이미지에 다른점이 너무 많아짐 적당히 조절 필요.
            # 두 이미지의 픽셀이 같으면 해당 위치는 하얀색 다르면 검정색
            for i in range(h):
                for j in range(w):
                    if t_img[i][j] == c_img[i][j]:
                        n_img[i][j] = 255
                    else:
                        n_img[i][j] = 0
            n_img = cv2.dilate(n_img,k,iterations=4)
            # 이미지 이진화
            _, n_img = cv2.threshold(n_img, 0, 255, cv2.THRESH_OTSU)
            cv2.imshow("test",n_img)
            cv2.waitKey()
            # 이미지에서 윤곽선 찾기
            conts, hier = cv2.findContours(n_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            n_img = cv2.cvtColor(n_img, cv2.COLOR_GRAY2RGB)
            # 찾은 윤곽선에 임의의 임계보다 큰 물체가 있다면 두 이미지는 다른 이미지임
            # for cont in conts:
            #     x, y, w, h = cv2.boundingRect(cont)
            #     if not x and not y:
            #         continue
            #     if w > 15 and h > 25:
            #         cv2.rectangle(n_img, (x,y), (x+w, y+h), (0,0,255))     
            #         cnt += 1
            #     cv2.imshow("nimg", n_img)  
            # 음표 이미지 분류
            if cnt == 0:
            #     print(t[:2])
            #     p += os.listdir(r"Classific_Symbols/8b/")+\
            #             os.listdir(r"Classific_Symbols/8/")+\
            #             os.listdir(r"Classific_Symbols/4/")+\
            #             os.listdir(r"Classific_Symbols/2/")+\
            #             os.listdir(r"Classific_Symbols/1/")+\
            #             os.listdir(r"Classific_Symbols/16/")+\
            #             os.listdir(r"Classific_Symbols/16b/")
            #     print(p)
            #     if c not in p:
            #         if "16" in t:
            #             if "b" in t:
            #                 cv2.imwrite(r"Classific_Symbols/16b/"+c,c_img)
            #             else:
            #                 cv2.imwrite(r"Classific_Symbols/16/"+c,c_img)
            #         elif "8" in t:
            #             if 'b' in t:
            #                 cv2.imwrite(r"Classific_Symbols/8b/"+c,c_img)
            #             else:
            #                 cv2.imwrite(r"Classific_Symbols/8/"+c,c_img)
            #         elif '4' in t:
            #             cv2.imwrite(r"Classific_Symbols/4/"+c,c_img)
            #         elif '2' in t:
            #             cv2.imwrite(r"Classific_Symbols/2/"+c,c_img)
            #         elif '1' in t:
            #             cv2.imwrite(r"Classific_Symbols/1/"+c,c_img)
                # cv2.imshow("img", img)
                cv2.moveWindow("test", 300, 300)
        
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
            print('O_none')
            return
        
        R_sifts = getPoint(Rt_img, Rc_img)
        if not R_sifts:
            print('R_none')
            return

        O_sift_matches = self.__sift_matcher.match(*O_sifts[1])
        R_sift_matches = self.__sift_matcher.match(*R_sifts[1])

        flag = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

        i = 6

        print(f'O_match : {len(O_sift_matches)}, R_match : {len(R_sift_matches)}')
        # img = cv2.drawMatches(Rt_img, R_sifts[0][0], Rc_img, R_sifts[0][1], O_sift_matches, None, flags=flag)
        # cv2.imshow('test',img)
        if len(O_sift_matches) > 4:
            G_O_Match = GoodMatch(O_sift_matches, O_sifts[0])
            print(f'SIFT Total Matching : {len(O_sift_matches)}, Good Matching : {len(G_O_Match)}')
            if len(G_O_Match) > len(O_sift_matches)//2:
                # print(f'SIFT Total Matching : {len(O_sift_matches)}, Good Matching : {len(G_O_Match)}')
                # Ot_img = cv2.resize(Ot_img,(Oc_img.shape[1],Oc_img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
                # _, Ot_img = cv2.threshold(Ot_img,0,255,cv2.THRESH_OTSU)
                # 올바른 매칭점을 이용해 매칭점 연결하기
                img = cv2.drawMatches(Ot_img, O_sifts[0][0], Oc_img, O_sifts[0][1], G_O_Match, None, flags=flag)
                cv2.imshow("o_img",img)
                # Classification(Ot_img, Oc_img, it=i)

        if len(R_sift_matches) > 4:
            G_R_Match = GoodMatch(R_sift_matches, R_sifts[0])
            print(f'SIFT Total Matching : {len(R_sift_matches)}, Good Matching : {len(G_R_Match)}')
            if len(G_R_Match) > len(R_sift_matches)//2:
                # print(f'SIFT Total Matching : {len(R_sift_matches)}, Good Matching : {len(G_R_Match)}')
                # Rt_img = cv2.resize(Rt_img,(Rc_img.shape[1],Rc_img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
                # _, Rt_img = cv2.threshold(Rt_img,0,255,cv2.THRESH_OTSU)
                # 올바른 매칭점을 이용해 매칭점 연결하기
                img = cv2.drawMatches(Rt_img, R_sifts[0][0], Rc_img, R_sifts[0][1], G_R_Match, None, flags=flag)
                cv2.imshow("r_img",img)
                # Classification(Rt_img, Rc_img, it=i)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()


    def GetCorner(self, img):
        
        ## 코너 추출(img,maxCorner,qualityLevel,minDistance)
        corners = cv2.goodFeaturesToTrack(img,20,0.001,3)

        nor = []

        for cor in corners:
            x, y = [*map(int, *cor)]
            nor.append((x, y))
        
        return nor

    ## 음표 분류를 위해 NHSW 사용해보기
    def compare(self, img1, img2):
        ###############################################################################################


        def calcVector(src, feature, patchSize):
            ## 특징 벡터 저장 리스트
            featureVectors = []
            for x, y in feature:
                ## 이미지의 특징 좌표를 중심으로 8x8 크기의 이미지 패치 추출
                featurePatch = src[y-patchSize//2:y+patchSize//2, x-patchSize//2:x+patchSize//2]

                ## 추출한 패치 내부의 모든 픽셀 값을 특징 벡터로 변환
                featureVector = featurePatch.reshape(-1)
                ## 값의 정규화
                featureVector = featureVector / 255
                
                ## 각 특징점의 특징 벡터를 모두 추합해 하나의 특징벡터로 변환
                featureVectors.extend(featureVector)
            return featureVectors


        def createVector(src, patchSize):
            ## 이미지 이진화
            _, src = cv2.threshold(src,128,255,cv2.THRESH_OTSU)

            ## 이미지 패딩 추가
            src = np.pad(src, ((patchSize,patchSize),(patchSize,patchSize)),mode='maximum')

            return calcVector(src,self.GetCorner(src),patchSize)
        ## HNSW 예시

        ## GFTT 코너 추출 알고리즘을 이용하여 각 음표 이미지의 코너를 추출하고 <- 이미지의 크기가 미치는 영향 분석 or 이미지들을 여러 크기로 변경해가며 추출
        #### -> 이미지가 크면 코너가 많이 추출됨 20개에 가깝게 추출되는 크기를 선정
        ## 추출된 특징 좌표를 좌표 주변 영역을 이용해 특징 벡터를 생성 <- 이미지의 이진화 알고리즘의 적용 전 후 차이 또는 영향 분석
        #### -> 이진화를 적용하지 않으면 코너 추출시 잡음 많이 발생
        ## 생성된 특징 벡터를 HNSW 알고리즘에 접목 시켜 유사도 측정
        ## 이때, HNSW 의 인덱싱 이미지가 많을수록 정확

        import faiss

        path = 'Kaggle_Music_note_datasets/datasets/datasets/Notes/'
        folderList = os.listdir(path)

        ## 입력 데이터 경로
        inputPath = 'Find_Symbols/'
        ## 입력 데이터 리스트
        inputList = os.listdir(inputPath)
        ## 데이터 리스트 정렬
        inputList = sorted(inputList, key = lambda x : int(x.split('.')[0]))

        ## 데이터셋 리스트
        dataSet = []

        for folderName in folderList:
            ## 데이터 셋 경로
            dataPath = path + folderName +'/'
            # dataPath = 'template/'
            ## 데이터 셋 리스트
            dataList = os.listdir(dataPath)
            ## 데이터 리스트 정렬
            dataList = sorted(dataList, key = lambda x : int(x.split('.')[0][1:]))
            # dataList = sorted(dataList, key = lambda x : int(x.replace('_','').replace('b','').split('.')[0]))
            dataList = [*map(lambda x : folderName + '/' + x,dataList)]
            dataSet.extend(dataList)

        ## 영상 코너 특징점은 최대 20개까지만 추출
        ## 추출된 특징점을 주변의 패치 영역을 추출하고 특징 벡터로 전환 후 HNSW에 입력
        def addingData_Case1():
            ## 각 이미지 마다의 특징벡터가 저장됨
            vectorImages = []

            ## 특징 벡터 저장 리스트
            queryFeatureVectors = []

            
            inputImage = cv2.imread(path+dataSet[0],cv2.IMREAD_GRAYSCALE)
            inputImage = cv2.resize(inputImage,dsize=None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)

            ## 벡터 추출 패치 사이즈
            # 이미지 크기의 10% => 추출에 실패하는 경우 발생
            patchSize = inputImage.shape[0] // 10

            ## 이미지 읽기
            for iName in dataSet:
                ## 데이터 이미지 입력
                inputImage = cv2.imread(path + iName, cv2.IMREAD_GRAYSCALE)
                ## 이미지 크기 변경
                inputImage = cv2.resize(inputImage,dsize=None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)

                vectorImages.append(createVector(inputImage,patchSize))

            ## HNSW 인덱스 생성
            dim = len(vectorImages[22]) ## 특징 벡터 차원

            featureIndex = faiss.IndexHNSWFlat(dim, 32)
            
            vectorImages = np.array(vectorImages)
                    
            ## HNSW 데이터 학습 및 추가
            featureIndex.train(vectorImages)
            featureIndex.add(vectorImages)

            ## 쿼리 이미지 입력
            queryImage = cv2.imread(inputPath + inputList[22], cv2.IMREAD_GRAYSCALE)

            ## 이미지 크기를 데이터셋 이미지 크기로 확장
            queryImage = cv2.resize(queryImage, dsize=tuple(reversed(inputImage.shape)),interpolation=cv2.INTER_CUBIC)

            ## 추합된 하나의 이미지의 특징 벡터를 저장
            queryFeatureVectors = np.array([createVector(queryImage,patchSize)])

            ### 쿼리 수행
            k = 3 # 가장 유사한 이웃 수
            D, I = featureIndex.search(queryFeatureVectors, k)

            similarity_scores = 1 / (1 + D)
            D = [d for d in D]
            similarity_scores = [s * 1000 for s in similarity_scores]

            print('최근접 이웃 거리:', *D)
            print('최근접 이웃 유사도 점수:', *similarity_scores)
            print('최근접 이웃 인덱스:', *I)

        addingData_Case1()


## 비슷하게 생겨서 잘못 인식된 이미지들 분류
## 위 결과 확인하면서 임계값 조정 하기

if __name__ == "__main__":
    cla = Classification()
    # cla.delete_Noise("template/")
    # cla.Standardization("template/", "Find_Symbols/")
    t_imgs = sorted(os.listdir("template/"),key=lambda x : int(x.replace(".jpg","").replace("_","").replace('b',"")))
    c_imgs = sorted(os.listdir("Find_Symbols/"),key=lambda x:int(x[:-4]))
    t_img = cv2.imread("template/8_1.jpg", cv2.IMREAD_GRAYSCALE)
    c_img = cv2.imread("Find_Symbols/22.jpg", cv2.IMREAD_GRAYSCALE)
    
    cla.compare(t_img,c_img)
    # cla.Resize([t_img, c_img], 9, cv2.INTER_LANCZOS4)
    # t_imgs = ["4_1.jpg","4_2.jpg"]
    # for t in t_imgs:
    #     t_img = cv2.imread("template/"+t, cv2.IMREAD_GRAYSCALE)
    #     for c in c_imgs:
    #         os.system('cls')
    #         print(t)
    #         print(c)
    #         c_img = cv2.imread("Find_Symbols/"+c, cv2.IMREAD_GRAYSCALE)
    #         h,w  = c_img.shape
    #         if w < 50:
    #             cla.Resize([t_img, c_img], 7, cv2.INTER_LANCZOS4)