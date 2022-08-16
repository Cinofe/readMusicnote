import cv2

class parent:
    # 이미지를 컬러 영상에서 -> 흑백 영상으로 변환
    def GrayScale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # 이미지 이진화
    def binary(self, img):
        _, new_img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        return new_img