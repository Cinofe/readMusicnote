import cv2, os, numpy as np


def Resize(img,Scale,interpol):
    src = img.copy()
    _, src = cv2.threshold(src,0,255,cv2.THRESH_OTSU)
    h, w = src.shape
    src = cv2.resize(src, (w*Scale, h*Scale), interpolation=interpol)
    return src

def Shi_TomsiCorner(img, type):
        h, w = img.shape
        # 
        corners = cv2.goodFeaturesToTrack(img,20,0.001,10)

        nor = []
        # 
        for cor in corners:
            x, y = [*map(int, *cor)]
            nor.append((x/w, y/h))

        nor.append(type)

        return nor

def GFTTDector(img, type):
    
    gftt = cv2.GFTTDetector_create()

    point = gftt.detect(img,None)

    h,w = img.shape

    point = [(p.pt[0]/w, p.pt[1]/h) for p in point]
    point.append(type)

    # draw = cv2.drawKeypoints(img, point, None)

    # cv2.imshow("abcd",draw)
    # cv2.waitKey()

    return point

def writing(f, datas):
    for data in datas:
        f.write(str(data))
        f.write(',')
    f.write('\n')


def main():
    img_path = r"Kaggle_Music_note_datasets/datasets/datasets/Notes/Eight/"
    imgs = os.listdir(img_path)
    with open('dataset.txt','w') as f:
        for i_name in imgs[:10]:
            img = cv2.imread(img_path+i_name,cv2.IMREAD_GRAYSCALE)
            
            img = Resize(img,5,cv2.INTER_CUBIC)
            # 5개
            # keypoints.append(Shi_TomsiCorner(img,"8"))
            keypoint = (GFTTDector(img,"8"))
            writing(f, keypoint)


if __name__ == "__main__":
    
    main()
    # file = np.genfromtxt("dataset.txt", delimiter=',')
    # print(file)
    # cvFile = file[:,:-1]
    # print(cvFile)
    # labelFile = file[:,-1]
    # print(labelFile)
    # cv = cvFile.astype(np.float32)
    # print(cv)
    # label = labelFile.astype(np.float32)
    # print(label)
    # knn = cv2.ml.KNearest_create()
    # knn.train(cv,cv2.ml.ROW_SAMPLE,label)
