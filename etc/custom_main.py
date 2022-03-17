import cv2, matplotlib.pyplot as plt

def line_pos():
    origin_img = cv2.imread(r'musicnotes/etc1.jpg')
    gray = cv2.cvtColor(origin_img,cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 1800 or gray.shape[1] > 1800:
        dst = cv2.resize(gray, dsize=(gray.shape[1]//3, gray.shape[0]//3), interpolation=cv2.INTER_AREA)
    else:
        dst = gray.copy()

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > 200:
                dst[i][j] = 255
            else:
                dst[i][j] = 0

    hist = []
    for i in range(dst.shape[0]):
        val = 0
        for j in range(dst.shape[1]):
            if dst[i][j] == 0:
                val += 1
        hist.append([i,val])

    lines = [val for val in hist if val[1] > dst.shape[1]//2]
    # print(len(lines)

    # cv2.imshow('test',dst)

    # plt.plot(hist)
    # plt.show()

    # cv2.waitKey()

    return dst, lines


def main():
    dst, lines = line_pos()

    print(lines)


    s = lines[0][0]

    for i in range(s,s+10):
        pass


if __name__ == '__main__':
    main()