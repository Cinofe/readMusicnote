# 템플릿 매칭으로 객체 위치 검출 (template_matching.py)

import cv2, os, numpy as np, shutil

def compare(src, template, iName, tName, type = 1, debug = False):
    if debug:
        print('=========================')
    for i in range(2,5):
        if type == 1:
            img = cv2.resize(src.copy(), dsize=None, fx=i,fy=i, interpolation=cv2.INTER_CUBIC)
        else : img = src.copy()

        template2 = template.copy()
        
        h, w = img.shape
        th, tw = template2.shape
        ## 이미지 비율 차이가 극심할 경우 올바른 비교가 아니므로 false 반환해야 함
        if th > h:
            ratio = h/th
            if ratio < 0.2:
                return False
            template2 = cv2.resize(template2, dsize=None, fx=h/th, fy=h/th,interpolation=cv2.INTER_AREA)
        th, tw = template2.shape
        if tw > w:
            ratio = w/tw
            if ratio < 0.2:
                return False
            template2 = cv2.resize(template2, dsize=None, fx=w/tw, fy=w/tw,interpolation=cv2.INTER_AREA)

        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        _, template2 = cv2.threshold(template2, 127, 255, cv2.THRESH_OTSU)

        # matchTemplate 함수를 사용하여 템플릿 이미지와 일치하는 부분 찾기
        # cv2.TM_CCOEFF_NORMED : 가장 성능이 뛰어나지만 수식이 복잡해서 연산량이 많습니다.
        # 완전히 일치하면 1, 역일치하면 -1, 상호 연관성이 없으면 0을 반환합니다.
        result = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)

        # result 값이 가장 큰 위치를 찾아냅니다.
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        img2 = cv2.rectangle(img.copy(), (x,y),(x+tw, y +th),(0,0,255),1)
        
        if debug:
            print(f'{iName}, {tName} : {max_val}')

            cv2.imshow('Original Image',img)
            cv2.imshow('Template',template)
            cv2.imshow('Resize template',template2)
            cv2.imshow('ROI 1',img2)

            cv2.moveWindow('Original Image',100,100)
            cv2.moveWindow('Template',250,100)
            cv2.moveWindow('Resize template', 400, 100)
            cv2.moveWindow('ROI 1',550,100)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # result2 값이 일정 threshold 이상인 경우에는 템플릿 이미지와 일치한다고 판단
        ## 전체 데이터 셋 84개
        ## 기준 값 0.605 일 경우 오류율 : 16.6%
        ## 기준 값 0.55 일 경우 오류율 : 4.76%
        ## 기준 값 0.57 일 경우 오류율 : 1.1%
        ## 전체 데이터 셋 259개, 
        # 문제점 1. 탬플릿이 명확하지 않음 -> 데이터 추가 완료
        # 문제점 2. 추출된 음표가 더러움 -> 수정 완료
        ## 기준 값 0.55 일 경우 오류율 : 7.7%
        ## 기준 값 0.57 일 경우 오류율 : 5.8%
        ## 기준 값 0.58 일 경우 오류율 : 6.9%
        ## 기준 값 0.6 일 경우 오류율 : 6.9%
        if type == 1:
            if max_val > 0.57:
                return True
        elif type == 2:
            if max_val > 0.40:
                return True
            break
    return False

## 2차 검증에서 크기 변환 과정이 필요한가? 괜히 잘못된 결과만 반환하는게 아닌가?
def secondCompare(src, templates, iName, tName, debug = False):
    outs = []
    for template in templates:
        template = cv2.imread(originTemplatePath+template, cv2.IMREAD_GRAYSCALE)
        outs.append(compare(src,template,iName,tName,type = 2, debug = debug))
    
    return sum(outs)

def writeImage(path : str, img):
    if os.path.exists(path):
        os.remove(path)
    cv2.imwrite(path, img)
    path = path.split('/')[-2]
    print(f'img : {iName}, target : {path}')

def clearFolder():
    path = r'./result/'
    fList = os.listdir(path)
    for fName in fList:
        if os.path.exists(path+fName):
            shutil.rmtree(path+fName)
        os.mkdir(path+fName)

clearFolder()
# 입력이미지와 템플릿 이미지 읽기
imgPath = 'testData/'
partTemplatePath = 'Sample/part/'
originTemplatePath = 'Sample/origin/'

## 입력 이미지 리스트 불러오기
imgList = os.listdir(imgPath)
## 이미지 이름 오름차순 정렬
imgList = sorted(imgList, key = lambda x : int(x.split('.')[0]))
## 부분 템플릿 이미지 리스트 불러오기
partTemplateList = os.listdir(partTemplatePath)
## 이미지 이름 오름차순 정렬
partTemplateList = sorted(partTemplateList, key= lambda x : int(x.split('.')[0]))
## 전체 템플릿 이미지 리스트 불러오기
originTemplateList = os.listdir(originTemplatePath)
## 이미지 이름 오름차순 정렬
originTemplateList = sorted(originTemplateList, key= lambda x : int(x.split('.')[0]))
# types = dict(map(lambda x : (int(x.split('.')[0])-1,False), originTemplateList))

## 1. 빔이 달려 있는가 ? T : beam, F : 2.
## 2. 꼬리가 달려 있는가 ? T : 8 or 16, F : 3.
## 3. 대가 달려 있는가 ? T : 4-1., F : 4-2.
## 4-1. 헤드에 구멍이 있는가 ? T : 2, F : 4
## 4-2. 헤드에 구멍이 있는가 ? T : 1, F : noise
## 각 이미지 마다 types 측정 후 특정 타겟에 만족하면 해당 type으로 지정
for iName in imgList:
    print(iName)
    # 0, 1, 2 : 빔, 3, 4: 8 꼬리, 5 : 16 꼬리, 6: 대, 7, 8 : 2분 헤드, 9: 4분 헤드
    types = {0:False, 1:False, 2:False, 3:False, 4:False, 5:False, 6:False, 7:False, 8:False, 9:False}
    img = cv2.imread(imgPath + iName, cv2.IMREAD_GRAYSCALE)
    for i, templateName in enumerate(partTemplateList):
        template = cv2.imread(partTemplatePath+templateName, cv2.IMREAD_GRAYSCALE)
        # print(iName, templateName, compare(img, template))
        types[i] = compare(img, template, iName, templateName)

    Type = [*types.values()]

    print(f'img : {iName}, result : {Type}')
    ## 1차 검증 이후에 음표 원본 그대로 2차 템플릿 비교 수행
    ## 2차 비교에 쓸 템플릿 이미지가 좀 더 다양해야함.
    ## 1차 역시 좀 더 다양하면 정확도 높아질 듯 함
    debug = True

    if sum(Type[0:3]) and Type[6] and sum(Type[7:]):
        if secondCompare(img, originTemplateList[-2:], iName, templateName, debug):
            ## 빔 음표
            writeImage(f'result/b/{iName}', img)
        else : writeImage(f'result/noise/{iName}', img)
    elif sum(Type[3:6]) and sum(Type[7:]):
        if secondCompare(img, originTemplateList[4:6], iName, templateName, False):
            ## 8분 또는 16분 음표 
            writeImage(f'result/8/{iName}', img)
        else : writeImage(f'result/noise/{iName}', img)
    elif Type[6] and Type[9]:
        if secondCompare(img, originTemplateList[:2], iName, templateName, False):
            ## 4분 음표
            writeImage(f'result/4/{iName}', img)
        else : writeImage(f'result/noise/{iName}', img)
    elif Type[6] and sum(Type[7:9]):
        ## 2분 음표
        writeImage(f'result/2/{iName}', img)
    elif sum(Type[7:9]):
        ## 온 음표
        writeImage(f'result/1/{iName}', img)
    else:
        ## 잡음
        writeImage(f'result/noise/{iName}', img)