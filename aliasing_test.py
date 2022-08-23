import cv2, numpy as np, matplotlib.pyplot as plt
from Resize import Resize

# 원본
src = cv2.imread(r'Find_Symbols/73.jpg', 0)
_,src = cv2.threshold(src,127,255,cv2.THRESH_OTSU)
dst = src.copy()
w,h = src.shape

## 경계선 추출
Lap_src = cv2.Laplacian(src,-1)
Lap_src = (255)-Lap_src
_,Lap_src = cv2.threshold(Lap_src,127,255,cv2.THRESH_OTSU)

## 경계선 좌표 추출
pos = []
for y,i in enumerate(Lap_src):
    pos.extend([(x,y) for x, v in enumerate(i) if v == 0])
## 지정 방향 탐색
Os = {'O1':[(0,-2),(0,-1),(0,0),(0,1),(0,2)],
    'O2':[(2,-2),(1,-1),(0,0),(-1,1),(-2,2)],
    'O3':[(2,0),(1,0),(0,0),(-1,0),(-2,0)],
    'O4':[(2,2),(1,1),(0,0),(-1,-1),(-2,-2)]}

for x, y in pos:
    O_value = {'O1':[],'O2':[],'O3':[],'O4':[]}
    for O in Os:
        cnts= []
        for Ox, Oy in Os.get(O):
            O_value.get(O).append(1 if src[y+Oy][x+Ox] == 0 else 0)
        cnts.append(O_value.get(O).count(1))
        print(cnts)
        if 1 in cnts:
            dst[y][x] = 255
            break
        elif 5 in cnts:
            break
        else :
            for cnt in cnts:
                if cnt < 3:
                    v = O_value.get(O)
                    for i in range(len(v)-2):                
                        if v[i] == 1 and v[i+1] == 0 and v[i+2] == 1:
                            ox,oy = Os.get(O)[i+1]
                            dst[y+oy][x+ox] = 0
                            break
                    break
            
        
        # merged = np.hstack([src,dst,Lap_src])
        # plt.figure(figsize=(17,6))
        # plt.imshow(merged)
        # plt.scatter(x, y, s=30, c="w", alpha=0.7)
        # x -= 0.5
        # y -= 0.5
        # v2 = [-2,-1,0,1,2,3]
        # for v in [-2,-1,0,1,2,3]:
        #     plt.plot([x-2,x+3],[y+v,y+v],color='red')
        #     plt.plot([x+v,x+v],[y-2,y+3],color='red')
        # plt.show()

merged = np.hstack([src,dst])
plt.figure(figsize=(17,6))
plt.imshow(merged)
# plt.scatter(y, x, s=30, c="w", alpha=0.7)
plt.show()

# ## 코너 탐지 테스트
# src = cv2.imread(r'Find_Symbols/73.jpg', 0)
# h,w = src.shape
# _,src = cv2.threshold(src,127,255,cv2.THRESH_OTSU)
# src2 = src.copy()
# src3 = src.copy()
# src4 = src.copy()

# dst1 = cv2.cornerHarris(src, 2, 3, 0.01)
# dst2 = cv2.cornerHarris(src2, 2, 3, 0.02)
# dst3 = cv2.cornerHarris(src3, 3, 3, 0.04)
# dst4 = cv2.cornerHarris(src4, 3, 3, 0.05)
# # thresholding
# _, dst1 = cv2.threshold(dst1, 0.001 * dst1.max(), 1, 0)
# _, dst2 = cv2.threshold(dst2, 0.001 * dst2.max(), 1, 0)
# _, dst3 = cv2.threshold(dst3, 0.001 * dst3.max(), 1, 0)
# _, dst4 = cv2.threshold(dst4, 0.001 * dst4.max(), 1, 0)

# x1, y1 = np.nonzero(dst1)
# x2, y2 = np.nonzero(dst2)
# x3, y3 = np.nonzero(dst3)
# x4, y4 = np.nonzero(dst4)

# merged = np.hstack([src,src2,src3,src4])

# plt.figure(figsize=(7,7))
# plt.title("Harris Corner Dectection 결과")
# plt.axis("off")
# plt.imshow(merged, cmap="gray")
# plt.scatter(y1, x1, s=30, c="w", alpha=0.7)
# plt.scatter(y2+(h-2), x2, s=30, c="w", alpha=0.7)
# plt.scatter(y3+((h-2)*2), x3, s=30, c="w", alpha=0.7)
# plt.scatter(y4+((h-2)*3), x4, s=30, c="w", alpha=0.7)

# plt.show()
