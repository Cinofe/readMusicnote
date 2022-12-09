import cv2, numpy as np
from matplotlib import pyplot as plt
from antiAliasing import antiAliasing
from Resize import Resize

Re = Resize()
ali = antiAliasing()

imgs = []

template_src = cv2.imread(r'template_img/Normal/8_Score1.jpg')
template_src = cv2.cvtColor(template_src, cv2.COLOR_RGB2GRAY)
compare_src = cv2.imread(r'Find_Symbols/130.jpg')
compare_src = cv2.cvtColor(compare_src, cv2.COLOR_RGB2GRAY)
print(f'template : {template_src.shape}, compare : {compare_src.shape}')

## create ORB
orb = cv2.ORB_create(nfeatures=400)

## create SIFT
sift = cv2.xfeatures2d.SIFT_create(nfeatures=400)

## create Matcher
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def Matching(datas):

    t_src, c_src = datas

    t_src = cv2.resize(t_src, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    c_src = cv2.resize(c_src, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # binaryization
    _, t_src = cv2.threshold(t_src, 127, 255, cv2.THRESH_OTSU)
    _, c_src = cv2.threshold(c_src, 127, 255, cv2.THRESH_OTSU)

    # orb extracting
    kp1, des1 = orb.detectAndCompute(t_src, None)
    kp2, des2 = orb.detectAndCompute(c_src, None)

    orbs = [(kp1, kp2), (des1, des2)]
    # sift extracting
    kp1, des1 = sift.detectAndCompute(t_src, None)
    kp2, des2 = sift.detectAndCompute(c_src, None)

    sifts = [(kp1, kp2), (des1, des2)]

    # orb Matching
    orb_matches = bf_orb.match(*orbs[1])

    # sift matching
    sift_matches = bf_sift.match(*sifts[1])
    
    # draw Matcher
    M_img_orb = cv2.drawMatches(t_src, orbs[0][0], c_src, orbs[0][1], orb_matches, None, flags=2)
    M_img_sift = cv2.drawMatches(t_src, sifts[0][0], c_src, sifts[0][1], sift_matches, None, flags=2)

    # img Combine
    combi_img = cv2.vconcat([M_img_orb, M_img_sift])
    combi_img = cv2.putText(combi_img, "ORB",(10,20),1,1.3,(0,0,0),1)
    combi_img = cv2.putText(combi_img, "SIFT",(10,276),1,1.3,(0,0,0),1)

    return combi_img

## template와 compare 이미지를 원본 크기부터 (256,256) size 까지 16크기 cv2.resize를 한 후 ORB, SIFT 매칭
## template의 크기만큼 compare 이미지를 점차 확대 시켜 ORB, SIFT 매칭

def S2B(temp, comp):
    t_h, t_w = temp.shape
    c_h, c_w = comp.shape
    # resizing
    src = comp.copy()
    _, src = cv2.threshold(src, 127, 255, cv2.THRESH_OTSU)
    ph = (t_h - c_h)/25
    pw = (t_w - c_w)/25
    print(ph,pw)
    for _ in range(26):
        h, w = src.shape
        src = cv2.resize(src,dsize=(int(w+pw),int(h+ph)),interpolation=cv2.INTER_CUBIC)

    return Matching([temp, src])

## temoplate와 compare 이미지를 한번에 (256,256) 크기로 cv2.resize를 적용한 후 ORB,SIFT 매칭
def O2B(imgs):
    datas = []
    for img in imgs:
        src = img.copy()
        _, src = cv2.threshold(src, 127, 255, cv2.THRESH_OTSU)
        src = cv2.resize(src,dsize=None, fx=5, fy=5,interpolation=cv2.INTER_CUBIC)
        datas.append(src)

    return Matching(datas)

img1 = S2B(template_src, compare_src)

# img2 = S2B([ali.antiAliasing(img) for img in imgs])

# img3 = O2B(imgs)

# img4 = O2B([ali.antiAliasing(img) for img in imgs])

# combi_img = cv2.hconcat([img1, img3])
plt.imshow(img1)
plt.show()
cv2.imshow('test',img1)
cv2.waitKey()

####
# 문제 발생, 특정 이미지에 대해서만 실험하여 다른 이미지에서 결과가 많이 일그러지게 됨
# 절대 좌표 형식으로 resize 하지 말고 비율로 resize 해봐야 할 듯 하다.
####

# ## Gradient
# def Gradient(d1,d2):
#     x1, y1 = d1
#     x2, y2 = d2
#     if (y2-y1) == 0 and (x2-x1) == 0:
#         return 0
#     elif (y2 - y1) == 0:
#         return 1/(x2-x1)
#     elif (x2 - x1) == 0:
#         return (y2 - y1)/1
#     else:
#         return (y2 - y1)/(x2 - x1)

# goodMat = []
# for i, mat in enumerate(orb_matches):
#     grad = Gradient(list(map(int, src1_kp_orb[mat.queryIdx].pt)), list(map(int, src2_kp_orb[mat.trainIdx].pt)))
#     if abs(grad) <= 3:
#         goodMat.append(mat)

# sift_matches = bf_sift.match(src1_des_sift, src2_des_sift)

# goodsift = []
# for mat in sift_matches:
#     grad = Gradient(list(map(int, src1_kp_sift[mat.queryIdx].pt)), list(map(int, src2_kp_sift[mat.trainIdx].pt)))
#     if abs(grad) < 3:
#         goodsift.append(mat)

# ## 키 포인트 제대로 매칭 됨
# for mat in orb_matches:
#     print(f'p1 : {[*map(int,src1_kp_orb[mat.queryIdx].pt)]}, p2 : {[*map(int,src2_kp_orb[mat.trainIdx].pt)]}')
#     x1, y1 = [*map(int,src1_kp_orb[mat.queryIdx].pt)]
#     x2, y2 = [*map(int,src2_kp_orb[mat.trainIdx].pt)]
#     cv2.line(M_img_orb,(x1,y1),(x2+160,y2),(0,255,0),2)
