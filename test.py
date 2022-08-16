import cv2, numpy as np
from matplotlib import pyplot as plt

def show_img(imgs):
    shows = []
    fig = plt.figure()
    row, col = 1, len(imgs)

    for i in range(col):
        shows.append(fig.add_subplot(row,col,i+1))
        shows[i].set_title(f'img {i+1}')
        shows[i].imshow(imgs[i], cmap=plt.cm.gray)
        shows[i].axis('off')

k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

src = cv2.imread(r'Find_Symbols/73.jpg', 0)
src = 255 - src

src = cv2.dilate(src, k_d, iterations=1)
src = cv2.erode(src, k_e, iterations=1)

src = 255 - src

res1 = cv2.resize(src, None, fx=2.0, fy= 2.0,interpolation=cv2.INTER_CUBIC)
# res1 = cv2.resize(src, (64,64),interpolation=cv2.INTER_CUBIC)
_, res1 = cv2.threshold(res1, 128, 255, cv2.THRESH_OTSU)

res2 = cv2.resize(res1, None, fx=2.0, fy= 2.0, interpolation=cv2.INTER_CUBIC)
# res2 = cv2.resize(res1, (128, 128), interpolation=cv2.INTER_CUBIC)
_, res2 = cv2.threshold(res2, 128, 255, cv2.THRESH_OTSU)

res3 = cv2.resize(res2, None, fx=2.0, fy= 2.0, interpolation=cv2.INTER_CUBIC)
# res3 = cv2.resize(res2, (256,256), interpolation=cv2.INTER_CUBIC)
_, res3 = cv2.threshold(res3, 128, 255, cv2.THRESH_OTSU)

# plt.imshow(res3, cmap=plt.cm.gray)

show_img([src,res1,res2,res3])

plt.show()