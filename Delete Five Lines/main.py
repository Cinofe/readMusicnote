import cv2, os
from module import *

def allimg():
    imgs = os.listdir(r'SheetMusics')
    for img in imgs:
        DFL = Del_FiveLine(img)
        whpos = list(zip(DFL.wpos,DFL.hist))
        DFL.delete_line(whpos)
        DFL.show(img)
        cv2.waitKey()
    
def oneimg(img_name):
    DFL = Del_FiveLine(img_name+".jpg")
    whpos = list(zip(DFL.wpos,DFL.hist))
    DFL.delete_line(whpos)
    DFL.show()
    cv2.waitKey()

oneimg('Sheet Music 2')
# allimg()
