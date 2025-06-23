# -*- coding: utf-8 -*-
# author: haroldchen0414

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from imutils import paths
import imutils
import numpy as np
import pickle
import progressbar
import random
import cv2
import os

class DocumentDenoiser:
    def __init__(self):
        self.basePath = "denoising-dirty-documents"
        self.testPath = os.path.join(self.basePath, "test")
        self.featuresPath = "features.csv"
        self.sampleProb = 0.02
        self.modelPath = "denoiser.pickle"

    # 利用中值模糊(5x5)获取图像的模糊版本
    # 计算前景(原始图像减去模糊图像)
    # 将前景值归一化到[0, 1]范围
    # 分母加上eps防止除0
    def blur_and_threshold(self, image, eps=1e-7):
        blur = cv2.medianBlur(image, 5)
        foreground = image.astype("float") - blur
        foreground[foreground > 0] = 0

        minVal = np.min(foreground)
        maxVal = np.max(foreground)
        foreground = (foreground - minVal) / ((maxVal - foreground) + eps)

        return foreground
    
    # 算法原理创建一个5x5的窗口从左到右, 从上到下滑动,取中心坐标(2, 2)为目标像素
    def build_features(self):
        trainPaths = sorted(list(paths.list_images(os.path.join(self.basePath, "train"))))
        cleanedPaths = sorted(list(paths.list_images(os.path.join(self.basePath, "train_cleaned"))))
        imagePaths = zip(trainPaths, cleanedPaths)
        
        widgets = ["创建特征:", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(trainPaths), widgets=widgets).start()

        with open(self.featuresPath, "w") as f:
            for(i, (trainPath, cleanedPath)) in enumerate(imagePaths):
                trainImage = cv2.imdecode(np.fromfile(trainPath, dtype=np.uint8), cv2.IMREAD_COLOR)
                cleanImage = cv2.imdecode(np.fromfile(cleanedPath, dtype=np.uint8), cv2.IMREAD_COLOR)
                trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
                cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)

                trainImage = cv2.copyMakeBorder(trainImage, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
                cleanImage = cv2.copyMakeBorder(cleanImage, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

                trainImage = self.blur_and_threshold(trainImage)
                cleanImage = cleanImage.astype("float") / 255.0

                for y in range(0, trainImage.shape[0]):
                    for x in range(0, trainImage.shape[1]):
                        trainROI = trainImage[y: y + 5, x: x + 5]
                        cleanROI = cleanImage[y: y + 5, x: x + 5]
                        (rH, rW) = trainROI.shape[:2]

                        if rW != 5 or rH != 5:
                            continue

                        features = trainROI.flatten()
                        target = cleanROI[2, 2]

                        if random.random() <= self.sampleProb:
                            features = [str(x) for x in features]
                            row = [str(target)] + features
                            row = ",".join(row)
                            f.write("{}\n".format(row))

                pbar.update(i)
        
        pbar.finish()

    def train_denoiser(self):
        self.build_features()

        features = []
        targets = []

        for row in open(self.featuresPath):
            row = row.strip().split(",")
            row = [float(x) for x in row]
            target = row[0]
            pixels = row[1:]
            features.append(pixels)
            targets.append(target)
        
        features = np.array(features, dtype="float")
        target = np.array(targets, dtype="float")

        (trainX, testX, trainY, testY) = train_test_split(features, target, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=10)
        model.fit(trainX, trainY)

        # 计算均方根误差
        preds = model.predict(testX)
        rmse = np.sqrt(mean_squared_error(testY, preds))
        print("rmse: {}".format(rmse))

        with open(self.modelPath, "wb") as f:
            f.write(pickle.dumps(model))

    def denoise_document(self, image_path):
        model = pickle.loads(open(self.modelPath, "rb").read())

        imagePaths = list(paths.list_images(image_path))
        random.shuffle(imagePaths)

        for imagePath in imagePaths[:5]:
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            orig = image.copy()

            image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
            image = self.blur_and_threshold(image)
            roiFeatures = []

            for y in range(0, image.shape[0]):
                for x in range(0, image.shape[1]):
                    roi = image[y: y + 5, x: x + 5]
                    (rH, rW) = roi.shape[:2]

                    if rW != 5 or rH != 5:
                        continue

                    features = roi.flatten()
                    roiFeatures.append(features)
            
            pixels = model.predict(roiFeatures)
            pixels = pixels.reshape(orig.shape)
            output = (pixels * 255).astype("uint8")

            cv2.imshow("Orig", imutils.resize(orig, width=600))
            cv2.imshow("Result", imutils.resize(output, width=600))
            cv2.waitKey(0)

denoiser = DocumentDenoiser()
denoiser.train_denoiser()
denoiser.denoise_document(os.path.join("sample"))