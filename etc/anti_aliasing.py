from math import nan
import cv2, numpy as np, matplotlib.pyplot as plt
from Resize import Resize

# 원본
src = cv2.imread(r'Find_Symbols/73.jpg', 0)

Re = Resize()

# 확대 이미지
re_src = Re.resizing(src,3)

# 흑백 이미지
_, bin_src = cv2.threshold(src, 127, 255, cv2.THRESH_OTSU)

# 경계 이미지
Lap_src = cv2.Laplacian(bin_src,-1)
Lap_src = (255) - Lap_src

# 원본 흑백 이미지에서 경계에 해당하는 부위에서만 
# 가우시안 블러를 이용해 픽셀간의 연속성을 늘린 뒤 계산
# x, y는 정수로, 해당하는 픽셀이 색상 값을 가져와야 하는 원본 이미지 크기의 흑백 이미지 좌표.

Gauss_src = cv2.GaussianBlur(bin_src, (0,0), 0.1)
h,w = Gauss_src.shape
M = []
for y in range(1,h-1):
    for x in range(1,w-1):
        if bin_src[y][x] == Lap_src[y][x]:
            print(x,y)
            gradx = (Gauss_src[y][x+1]-Gauss_src[y][x-1])/2
            grady = (Gauss_src[y+1][x]-Gauss_src[y-1][x])/2
            m = round((gradx/grady)*-1,2)
            M.append([x,y,m])

for x,y,m in M:
    if abs(m) == 1 :
        src[y][x] = 100

plt.imshow(src)
plt.show()


# gradx = ((x+1, y)-(x-1,y))/2
# grady = ((x,y+1)-(x,y-1))/2
# M = (gradx/grady)*-1


# cv2.imshow('src', src)
# cv2.imshow('re_src', re_src)
cv2.imshow('bin_src', bin_src)
cv2.imshow('Lap_src', Lap_src)
# cv2.imshow('Gauss',Gauss_src)

cv2.waitKey()