import cv2, os, time as t

def binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 220:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

def delete_line(img,wh):
    _,w = img.shape
    wp,hp = wh
    for i in range(wp,w):
        if img[hp-1,i] == 255:
            img[hp,i] = 255
        
    return img

def main():

    wpos = []
    times = []
    imgs = os.listdir(r'musicnotes')
    for img in imgs:
        t_start = t.time()
        fiveLine_del(wpos,img)
        t_end = t.time()
        times.append((img,t_end - t_start))
        t.sleep(1)
    # fiveLine_del(wpos,imgs[1])
    print(sum([time[1] for time in times])/8)

    cv2.waitKey()

def fiveLine_del(wpos,i):
    origin_img = cv2.imread(r'musicnotes/'+i)
    dst = cv2.cvtColor(origin_img,cv2.COLOR_BGR2GRAY)
    print(i,dst.shape)
    if dst.shape[0] > 1800 or dst.shape[1] > 1800:
        dst = cv2.resize(dst, dsize=(dst.shape[1]//3, dst.shape[0]//3), interpolation=cv2.INTER_AREA)

    # 이미지 이진화
    img = binary(dst)
    h,w = img.shape

    # cv2.imshow('origin',dst)
    # 대부분의 악보들의 오선 시작 위치는 악보의 너비 약 5% 지점에서 시작
    # 악보의 수직 히스토그램을 구하면 가장 오선의 y축 대비 위치를 쉽게 찾을 수 있음
    # 5% 지점은 오선의 시작 지점으로 임시 지정하고
    # 좌,우 측 픽셀값을 검사해가며 시작 위치를 추정
    # 이미지 좌측에서 가장 위에 있는 검정 픽셀 검출
    
    # 악보의 수평 히스토그램을 구함
    hist = find_hist(img, h, w)
    
    # 악보의 수직 히스토그램을 기반으로 오선의 시작 위치 추정
    wpos = findFiveLine(wpos, img, w, hist)
        
    # 검출된 검정 픽셀 선 삭제 
    whpos = list(zip(wpos,hist))
    for wh in whpos:
        img = delete_line(img, wh)

    ## 2022-01-18
    ## 잡음 제거 및 오선 제거하는 부분 추가적인 조건 필요
    ## 오선 이외의 빔 부분도 오선과 겹칠 경우 가끔 지워짐
    ## 개선 필요
    
    # cv2.imshow(i,img)

def findFiveLine(wpos, img, w, hist):
    s = (w//100)*5

    for h in hist:
        if img[h,s] == 0:
            p = 0
            while img[h,s-p] == 0:
                p += 1
            wpos.append((s-p)+1)
        else:
            p = 0
            while img[h,(s+p)] == 255:
                p += 1
            wpos.append((s+p)-1)
    
    return wpos

def find_hist(img, h, w):
    hist = []
    hist2 =[]
    y = []
    for i in range(1,h-1):
        value = 0
        for j in range(1,w-1):
            if img[i,j] == img[i,j-1]:
                weight = 1.2
            else:
                weight = 1
            if img[i,j] == 0:
                value += weight
        if value >= (w/100)*40 :
            hist.append(i)
        hist2.append(value)
        y.append(i)

    # plt.barh(y,hist2)
    # plt.show()
    return hist

if __name__ == '__main__':
    main()