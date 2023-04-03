from del_FiveLine import Del_FiveLine
from del_Noise import Del_Noise
import cv2, os

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

    DFL = Del_FiveLine(imgs[0])

    DN = Del_Noise(*DFL.GetImg(), DFL.hist)
    DN.delete_noise()
    DN.find_Contours()
    # DN.findData()


def main():
    oneimg()
    # allimg()


if __name__ == '__main__':
    main()
