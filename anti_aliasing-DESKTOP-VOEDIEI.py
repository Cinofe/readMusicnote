import cv2

src = cv2.imread(r'Find_Symbols/73.jpg', 0)

src_1 = cv2.resize(src,None,fx=5.0, fy=5.0,interpolation=None)
_, src_2 = cv2.threshold(src,128,255,cv2.THRESH_OTSU)

print(src.shape)
cv2.imshow('test1',src_1)
cv2.imshow('test2',src_2)
cv2.waitKey()