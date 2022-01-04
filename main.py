import cv2, numpy as np


def get_yh(img):
    yhs = []
    Line_kernel =cv2.getStructuringElement(cv2.MORPH_CROSS, (15,1))
    Rect_kernel =cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    _, Line_img = cv2.threshold(img,135,255,cv2.THRESH_BINARY)

    l_dilate = cv2.dilate(Line_img, Line_kernel, anchor=(-1,-1),iterations=5)
    l_erode = cv2.erode(l_dilate, Rect_kernel, anchor=(-1,-1),iterations=3)

    contours,_ = cv2.findContours(l_erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x,y,_,h = cv2.boundingRect(contour)
        if x > 150 or h > 100:
            continue
        y -= 20
        h += 40
        yhs.append((y,h))
    
    return yhs

if __name__ == '__main__':

    R_size = 384
    C_size = 512

    origin_img = cv2.imread(r'musicnotes/He_is_Pirate/pirate_1.jpg')
    # origin_img = cv2.imread(r'musicnotes/etc/etc1.jpg')

    origin_img = cv2.resize(origin_img, dsize=(R_size,C_size),interpolation=cv2.INTER_AREA)

    binary_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    yhs = get_yh(binary_img.copy())
    print(yhs)

    _, bimg = cv2.threshold(binary_img,155,255,cv2.THRESH_BINARY_INV)

    row_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    col_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,7))

    b_erode = cv2.erode(bimg, col_kernel, anchor=(-1,-1),iterations=1)
    b_dilate = cv2.dilate(b_erode, row_kernel, anchor=(-1,-1),iterations=1)
    

    contours, _ = cv2.findContours(b_erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for pts in contours:
        x,y,w,h = cv2.boundingRect(pts)
        if w > 100 or h > 100:
            continue
        
        dist = abs(x - (x+w))//4

        cv2.rectangle(origin_img, (x,y,w,h), (0,0,255), 1)
        x -= dist
        w += dist*2
        if y < 100 :
            y,h = yhs[3]
        elif y < 300 : 
            y,h = yhs[2]
        elif y < 450 :
            y,h = yhs[1]
        else:
            y,h = yhs[0]
        cv2.rectangle(origin_img, (x,y,w,h), (0, 0, 0), 1)
    
    result = np.concatenate((bimg,b_erode, b_dilate),axis=1)

    cv2.imshow('dilate and erode',result)
    cv2.imshow('origin',origin_img)

    cv2.waitKey(0)