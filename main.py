from class_Del_FiveLine import Del_FiveLine
from class_Del_Noise import Del_Noise
from Zhang_Suen_Thinning import Thinning
import cv2, os

def allimg():
    imgs = os.listdir(r'SheetMusics')
    for img in imgs:
        # t_start = t.time()
        DFL = Del_FiveLine(img)
        whpos = list(zip(DFL.wpos,DFL.hist))
        DFL.delete_line(whpos)
        # t_end = t.time()
        # print(f"img : {img}, img size(w,h) : {DFL.get_shape()}, process time : {t_end - t_start:.3f}sec")
        DFL.show(img)
        # DN = Del_Noise(*DFL.GetImg(),DFL.hist)
        # DN.find_degree(whpos)
        # DN.delete_noise()
        # DN.find_Contours()
        cv2.waitKey()
    
def oneimg():
    imgs = os.listdir(r'SheetMusics')
    DFL = Del_FiveLine(imgs[2])
    whpos = list(zip(DFL.wpos,DFL.hist))
    DFL.delete_line(whpos)
    # DFL.show()
    
    DN = Del_Noise(*DFL.GetImg(),DFL.hist)
    # DFL.find_degree(whpos)
    DN.delete_noise()
    DN.find_Contours()
    
    # TN = Thinning()
    # TN.Thinning()


    # DFL.show()

    cv2.waitKey()


def main():
    oneimg()
    # allimg()
    
if __name__ == '__main__':
    main()