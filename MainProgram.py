from class_Del_FiveLine import Del_FiveLine
from class_Del_Noise import Del_Noise
import cv2, time
import os


def allimg():
    imgs = os.listdir(r'SheetMusics')[10:20]
    for img in imgs:
        # t_start = t.time()
        DFL = Del_FiveLine(img)
        whpos = list(zip(DFL.wpos, DFL.hist))
        DFL.delete_line(whpos)
        # t_end = t.time()
        # print(f"img : {img}, img size(w,h) : {DFL.get_shape()}, process time : {t_end - t_start:.3f}sec")
        # DFL.show(img)
        DN = Del_Noise(*DFL.GetImg(), DFL.hist)
        # DN.find_degree(whpos)
        DN.delete_noise()
        DN.find_Contours()
        cv2.waitKey()


def oneimg():
    imgs = os.listdir(r'SheetMusics')
    ## img list

    imgs.sort(key=lambda x : int(x.replace(".jpg","").split()[-1]))

    DFL = Del_FiveLine(imgs[3])
    whpos = list(zip(DFL.wpos, DFL.hist))
    DFL.delete_line(whpos)
    # DFL.show()

    DN = Del_Noise(*DFL.GetImg(), DFL.hist)
    # DFL.find_degree(whpos)
    # start = time.time()
    DN.delete_noise()
    DN.find_Contours()
    # DN.findData()
    # end = time.tim10()
    # print(f'{end-start}')

    # DFL.show()

    cv2.waitKey()


def main():
    oneimg()
    # allimg()


if __name__ == '__main__':
    main()
