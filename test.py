### 이미지에서 외곽선 이미지만 추출하고
### 해당 이미지를 그대로 벡터화 하여 
### HNSW 입력으로 넣어보기


##### 무언가 잘못됨
##### 다른 이미지와 다른 방법에 대해 같은 결과만 반환 됨.

import numpy as np, faiss, cv2, os

inputVectors = []

# 이미지 로드
# img = cv2.imread('Kaggle_Music_note_datasets/datasets/datasets/Notes/Eight/e1.jpg', cv2.IMREAD_GRAYSCALE)

path = 'Kaggle_Music_note_datasets/datasets/datasets/Notes/'
folderList = os.listdir(path)
## 입력 데이터 경로
inputPath = 'Find_Symbols/'
## 입력 데이터 리스트
inputList = os.listdir(inputPath)
## 데이터 리스트 정렬
inputList = sorted(inputList, key = lambda x : int(x.split('.')[0]))

## 데이터셋 리스트
dataSet = []

for folderName in folderList:
    ## 데이터 셋 경로
    dataPath = path + folderName +'/'
    # dataPath = 'template/'
    ## 데이터 셋 리스트
    dataList = os.listdir(dataPath)
    ## 데이터 리스트 정렬
    dataList = sorted(dataList, key = lambda x : int(x.split('.')[0][1:]))
    # dataList = sorted(dataList, key = lambda x : int(x.replace('_','').replace('b','').split('.')[0]))
    dataList = [*map(lambda x : folderName + '/' + x,dataList)]
    dataSet.extend(dataList[:100])

for iName in dataSet:
    ## 데이터 이미지 입력
    inputImage = cv2.imread(path + iName, cv2.IMREAD_GRAYSCALE)
    # 이진화
    # _, inputImage = cv2.threshold(inputImage, 127, 255, cv2.THRESH_OTSU)
    ## 이미지 크기 변경
    inputImage = cv2.resize(inputImage,dsize=None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(inputImage, 100, 200)

    # edges = cv2.bitwise_not(edges)

    edges = edges.reshape(-1)

    edges = edges / 255

    inputVectors.append(edges)

inputVectors = np.array(inputVectors)
dim = len(inputVectors[0])

featureIndex = faiss.IndexHNSWFlat(dim, 32)

## HNSW 데이터 학습 및 추가
featureIndex.train(inputVectors)
featureIndex.add(inputVectors)

queryImage = cv2.imread(inputPath + inputList[4], cv2.IMREAD_GRAYSCALE)
# _, queryImage = cv2.threshold(queryImage, 127, 255, cv2.THRESH_OTSU)
queryImage = cv2.resize(queryImage, dsize=tuple(reversed(inputImage.shape)),interpolation=cv2.INTER_CUBIC)
edges = cv2.Canny(inputImage, 100, 200)

# edges = cv2.bitwise_not(edges)

edges = edges.reshape(-1)

edges = edges /255

queryVector = np.array([edges])
### 쿼리 수행
k = 3 # 가장 유사한 이웃 수
D, I = featureIndex.search(queryVector, k)

similarity_scores = 1 / (1 + D)
D = [d for d in D]
# similarity_scores = [s * 1000 for s in similarity_scores]

print('최근접 이웃 거리:', *D)
print('최근접 이웃 유사도 점수:', *similarity_scores)
print('최근접 이웃 인덱스:', *I)

cv2.imshow('query Image', queryImage)
cv2.imshow('result Image', cv2.imread(path + dataSet[I[0][0]]))
cv2.waitKey()
