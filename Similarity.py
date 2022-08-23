import cv2, numpy as np
from matplotlib import pyplot as plt
from Zhang_Suen_Thinning import Thinning

TN = Thinning()

src1 = cv2.imread(r'template_img/Normal/Beam_8_Score2.jpg')
src1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)
src2 = cv2.imread(r'Find_Symbols/73.jpg')
src2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)

print(src1.shape, src2.shape)

## img Resizing
src1 = cv2.resize(src1,(64,64),interpolation=cv2.INTER_AREA)
src2 = cv2.resize(src2,(64,64),interpolation=cv2.INTER_LINEAR)

for _ in range(6):
    h,w = src1.shape
    src1 = cv2.resize(src1,(w+16,h+16),interpolation=cv2.INTER_LINEAR)
    src2 = cv2.resize(src2,(w+16,h+16),interpolation=cv2.INTER_LINEAR)

print(src1.shape, src2.shape)

## binary
_, src1 = cv2.threshold(src1,127,255,cv2.THRESH_OTSU)
_, src2 = cv2.threshold(src2,127,255,cv2.THRESH_OTSU)

## create ORB
orb = cv2.ORB_create(nfeatures=400)
src1_kp_orb, src1_des_orb = orb.detectAndCompute(src1, None)
src2_kp_orb, src2_des_orb = orb.detectAndCompute(src2, None)

## create SIFT
sift = cv2.xfeatures2d.SIFT_create(nfeatures=400)
src1_kp_sift, src1_des_sift = sift.detectAndCompute(src1, None)
src2_kp_sift, src2_des_sift = sift.detectAndCompute(src2, None)

print(f'1. ORB - origin : {len(src1_kp_orb)}, detect : {len(src2_kp_orb)}')
print(f'2. SIFT - origin : {len(src1_kp_sift)}, detect : {len(src2_kp_sift)}')


## create Mathcer
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

## Gradient
def Gradient(d1,d2):
    x1, y1 = d1
    x2, y2 = d2
    if (y2-y1) == 0 and (x2-x1) == 0:
        return 0
    elif (y2 - y1) == 0:
        return 1/(x2-x1)
    elif (x2 - x1) == 0:
        return (y2 - y1)/1
    else:
        return (y2 - y1)/(x2 - x1)

## Matching
orb_matches = bf_orb.match(src1_des_orb, src2_des_orb)

goodMat = []
for i, mat in enumerate(orb_matches):
    grad = Gradient(list(map(int, src1_kp_orb[mat.queryIdx].pt)), list(map(int, src2_kp_orb[mat.trainIdx].pt)))
    if abs(grad) <= 3:
        goodMat.append(mat)

sift_matches = bf_sift.match(src1_des_sift, src2_des_sift)

goodsift = []
for mat in sift_matches:
    grad = Gradient(list(map(int, src1_kp_sift[mat.queryIdx].pt)), list(map(int, src2_kp_sift[mat.trainIdx].pt)))
    if abs(grad) < 3:
        goodsift.append(mat)

## draw Keypoints
# src1_orb = cv2.drawKeypoints(src1, src1_kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# src2_orb = cv2.drawKeypoints(src2, src2_kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# src1_sift = cv2.drawKeypoints(src1, src1_kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# src2_sift = cv2.drawKeypoints(src2, src2_kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# print(f'src1 : {len(src1_kp_sift)}, src2 : {len(src2_kp_sift)}')
# cv2.imshow('src1_sift', src1_sift)
# cv2.imshow('src2_sift', src2_sift)

## draw Matches
M_img_orb = cv2.drawMatches(src1, src1_kp_orb, src2, src2_kp_orb, goodMat, None, flags=2)
M_img_sift = cv2.drawMatches(src1, src1_kp_sift, src2, src2_kp_sift, sift_matches, None, flags=2)

## img Combine
combi_img = cv2.vconcat([M_img_orb, M_img_sift])
combi_img = cv2.putText(combi_img, "ORB",(10,30),1,1.5,(0,0,0),1)
combi_img = cv2.putText(combi_img, "SIFT",(10,286),1,1.5,(0,0,0),1)


plt.imshow(combi_img)
plt.show()
cv2.waitKey()