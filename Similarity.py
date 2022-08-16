import cv2, numpy as np
from matplotlib import pyplot as plt
from Zhang_Suen_Thinning import Thinning

TN = Thinning()

src1 = cv2.imread(r'template_img/Normal/Beam_8_Score2.jpg')
src1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)
src2 = cv2.imread(r'Find_Symbols/73.jpg')
src2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)

print(src1.shape, src2.shape)

src3 = src1.copy()
src3 = TN.Thinning(src3)
src4 = cv2.imread(r'thinning_Symbols/73.jpg')
src4 = cv2.cvtColor(src4, cv2.COLOR_RGB2GRAY)

## img Resizing
src1 = cv2.resize(src1, (256,256), interpolation=cv2.INTER_LINEAR)
src2 = cv2.resize(src2, (256,256), interpolation=cv2.INTER_LANCZOS4)
src3 = cv2.resize(src3, (256,256), interpolation=cv2.INTER_LINEAR)
src4 = cv2.resize(src4, (256,256), interpolation=cv2.INTER_LINEAR)

## Thinning
t_src1 = TN.Thinning(src1)
t_src2 = TN.Thinning(src2)

## create ORB
orb = cv2.ORB_create(nfeatures=400)
src1_kp_orb, src1_des_orb = orb.detectAndCompute(src1, None)
src2_kp_orb, src2_des_orb = orb.detectAndCompute(src2, None)

src3_kp_orb, src3_des_orb = orb.detectAndCompute(src3, None)
src4_kp_orb, src4_des_orb = orb.detectAndCompute(src4, None)

t_src1_kp_orb, t_src1_des_orb = orb.detectAndCompute(t_src1,None)
t_src2_kp_orb, t_src2_des_orb = orb.detectAndCompute(t_src2,None)

## create SIFT
sift = cv2.xfeatures2d.SIFT_create(nfeatures=400)
src1_kp_sift, src1_des_sift = sift.detectAndCompute(src1, None)
src2_kp_sift, src2_des_sift = sift.detectAndCompute(src2, None)


src3_kp_sift, src3_des_sift = sift.detectAndCompute(src3, None)
src4_kp_sift, src4_des_sift = sift.detectAndCompute(src4, None)

t_src1_kp_sift, t_src1_des_sift = sift.detectAndCompute(t_src1, None)
t_src2_kp_sift, t_src2_des_sift = sift.detectAndCompute(t_src2, None)



print(f'1. ORB - origin : {len(src1_kp_orb)}, detect : {len(src2_kp_orb)}')
print(f'2. SIFT - origin : {len(src1_kp_sift)}, detect : {len(src2_kp_sift)}')
print(f'3. Thin_ORB - origin : {len(t_src1_kp_orb)}, detect : {len(t_src2_kp_orb)}')
print(f'4. Thin_SIFT - origin : {len(t_src1_kp_sift)}, detect : {len(t_src2_kp_sift)}')
print(f'5. Thin2_orb - origin : {len(src3_kp_orb)}, detect : {len(src4_kp_orb)}')
print(f'6. Thin2_SIFT - origin : {len(src3_kp_sift)}, detect : {len(src4_kp_sift)}')

## create Mathcer
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


## Gradient
def Gradient(d1,d2):
    x1, y1 = d1
    x2, y2 = d2
    if (y2 - y1) == 0:
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

t_orb_matches = bf_orb.match(t_src1_des_orb, t_src2_des_orb)

sift_matches = bf_sift.match(src1_des_sift, src2_des_sift)
goodsift = []
for mat in sift_matches:
    grad = Gradient(list(map(int, src1_kp_sift[mat.queryIdx].pt)), list(map(int, src2_kp_sift[mat.trainIdx].pt)))
    if abs(grad) < 3:
        goodsift.append(mat)

t_sift_matches = bf_sift.match(t_src1_des_sift, t_src2_des_sift)

else_matches_orb = bf_orb.match(src3_des_orb, src4_des_orb)
else_matches_sift = bf_sift.match(src3_des_sift, src4_des_sift)

## draw Keypoints
# src1_orb = cv2.drawKeypoints(src1, src1_kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# src2_orb = cv2.drawKeypoints(src2, src2_kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

src1_sift = cv2.drawKeypoints(src1, src1_kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
src2_sift = cv2.drawKeypoints(src2, src2_kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f'src1 : {len(src1_kp_sift)}, src2 : {len(src2_kp_sift)}')
cv2.imshow('src1_sift', src1_sift)
cv2.imshow('src2_sift', src2_sift)

## draw Matches
M_img_orb = cv2.drawMatches(src1, src1_kp_orb, src2, src2_kp_orb, goodMat, None, flags=2)
M_img_sift = cv2.drawMatches(src1, src1_kp_sift, src2, src2_kp_sift, sift_matches, None, flags=2)
t_M_img_orb = cv2.drawMatches(t_src1, t_src1_kp_orb, t_src2, t_src2_kp_orb, t_orb_matches, None, flags=2)
t_M_img_sift = cv2.drawMatches(t_src1, t_src1_kp_sift, t_src2, t_src2_kp_sift, t_sift_matches, None, flags=2)
e_M_img_orb = cv2.drawMatches(src3, src3_kp_orb, src4, src4_kp_orb, else_matches_orb, None, flags=2)
e_M_img_sift = cv2.drawMatches(src3, src3_kp_sift, src4, src4_kp_sift, else_matches_sift, None, flags=2)

## img Combine
combi_img = cv2.vconcat([M_img_orb, M_img_sift])
combi_img = cv2.putText(combi_img, "ORB",(10,30),1,1.5,(0,0,0),1)
combi_img = cv2.putText(combi_img, "SIFT",(10,286),1,1.5,(0,0,0),1)

t_combi_img = cv2.vconcat([t_M_img_orb, t_M_img_sift])
t_combi_img = cv2.putText(t_combi_img, "Thinning ORB",(10,30),1,1.5,(0,0,0),1)
t_combi_img = cv2.putText(t_combi_img, "Thinning SIFT",(10,286),1,1.5,(0,0,0),1)

e_combi_img = cv2.vconcat([e_M_img_orb, e_M_img_sift])

combi_img = cv2.hconcat([combi_img, t_combi_img])
combi_img = cv2.hconcat([combi_img, e_combi_img])

plt.imshow(combi_img)
plt.show()
cv2.waitKey()