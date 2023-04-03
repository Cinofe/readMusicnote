# import cv2, numpy as np

# # print(cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# origin = np.array([[0,0,1,1,1],
#                    [0,0,0,1,1],
#                    [1,1,1,1,1],
#                    [1,1,0,1,1],
#                    [1,1,0,1,1]],
#                     np.uint8)
# e_kernal = np.array([[0,1,0],[0,0,0],[0,1,0]],np.uint8)
# d_kernal = np.array([[0,1,0],[0,0,0],[0,1,0]],np.uint8)
# des = cv2.erode(origin,e_kernal,iterations=1)
# # des = cv2.dilate(origin,d_kernal,iterations=1)

# print(des)


import cv2, matplotlib.pyplot as plt

img1 = cv2.imread('./etc/123.jpg')
img2 = cv2.imread('./etc/123.jpg')
merge = cv2.hconcat([img1, img2])
cv2.imshow('test',merge)
cv2.waitKey()
# print(img.shape)

# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# print(img.shape)

# _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# print(img.shape)
