import os
import cv2, time as t

def binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 200:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

def delete_line(img,i):
    h,w = img.shape
    spos = None
    epos = None
    cnt = 0

    for j in range(h):
        if img[j,i] == 0 and j != 0:
            s = j
            a = j
            while img[a,i] != 255:
                a += 1
                cnt += 1
            e = s + cnt
            spos = (i,s-1)
            epos = (i,e+1)
            break
    
    if spos != None and epos != None:
        for i in range(spos[0],w-1):
            if (img[spos[1],i] == 255 and img[epos[1],i] == 255
             and img[spos[1],i-1] == 255 and img[epos[1],i-1] == 255):
                img[spos[1]:epos[1],i] = 255
    else : 
        return 1, img
        
    return 0 ,img

def main():
    dirs = os.listdir(r'musicnotes')
    times = []
    a = []
    for name in dirs:
        # t_start = t.time()
        fiveLine_del(name,a)
        # t_end = t.time()
    #     times.append((name,t_end-t_start))
    # print(sum([time[1] for time in times])/8)
    print(f'avr = {sum(a)/len(a):.3f}%')

    cv2.waitKey()

def fiveLine_del(imgname,a):

    wpos = None
    origin_img = cv2.imread(r'musicnotes/'+imgname)
    gray = cv2.cvtColor(origin_img,cv2.COLOR_BGR2GRAY)

    if gray.shape[0] > 1800 or gray.shape[1] > 1800:
        dst = cv2.resize(gray, dsize=(gray.shape[1]//3, gray.shape[0]//3), interpolation=cv2.INTER_AREA)
    else:
        dst = gray.copy()

    # 이미지 이진화
    img = binary(dst)

    cv2.imshow('origin'+imgname,dst)

    # 이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    for i in range(img.shape[1]):
        for j in range(100,img.shape[0]):
            if img[j,i] == 0 and i != 0 and j != 0:
                wpos = i
                print(f'img : {imgname}, black pixel pos : {wpos}, img_width : {img.shape[1]}, start pos : {(p:=wpos/img.shape[1]*100):.3f}%')
                a.append(p)
                break
        if wpos != None:
            break
    # 검출된 검정 픽셀에서 + 5위치 까지 선 삭제 
    for j in range(5):
        for i in range(img.shape[0]):
            stop,img = delete_line(img, wpos+j)
            if stop == 1:
                break
    
    # cv2.imshow(imgname,img)

if __name__ == '__main__':
    main()