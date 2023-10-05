# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

class Calbration():
    def __init__(self, imagePath="", cols=2, rows=2, squareSize=1.0):
        # log
        self.log = logging.getLogger("Calbration")
#        self.log.addHandler(logging.StreamHandler())
#        self.log.setLevel(logging.DEBUG)

        self.formats = ["*.jpg", "*.png"] #画像フォーマット
        self.image = [] # 画像バッファ
        self.imageSize = None # 画像サイズ
        self.imagePath = Path(imagePath)# 画像ファイルパス
        self.setConfig(cols, rows, squareSize)# 列×行

        self.log.debug("initial Calb..")

    def setConfig(self, cols, rows, squareSize):
        self.log.debug("setConfig")
        self.patternSize = (cols, rows)# 画像サイズ

        self.patternPoints = np.zeros( (np.prod(self.patternSize), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
        self.patternPoints[:,:2] = np.indices(self.patternSize).T.reshape(-1, 2)
        self.patternPoints *= squareSize # 正方形の1辺のサイズ[cm]を設定

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)# 高精度化する際の閾値。cornerSubPixする際に何回やるか、目標精度を設定

    # image read from iamge folder
    def read(self):
        if( os.path.exists(self.imagePath) ):
            for fmt in self.formats:
                for path in self.imagePath.glob(fmt):
                    img = cv2.imread(str(path)) # 画像読込み
                    self.image.append([ path, img])

            if( len(self.image) > 0 ):
                self.log.debug("find image..." + str(len(self.image)))
                return True
            else:
                # error
                self.log.debug("Don't exist image file.")
                return False
        else:
            # error
            self.log.debug("Don't exist folder.")
            return False

    def calbration(self):

        if( not self.read()):# read image
            # error
            return False

        self.log.debug("corner finding start")
        imgPoints = []# corner buff
        objPoints = []# obj buff
        count = 0
        for img in self.image: 
            self.log.debug(str(img[0]) + " find...")

            gray = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
            if(self.imageSize is None):
                self.imageSize = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, self.patternSize, None)
            if(ret):
                self.log.debug("detected corner")

                corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria) #コーナー位置精度補正。(11,11)のところは高精度化する際の探索窓の大きさ？
                imgPoints.append(corners)
                objPoints.append(self.patternPoints)

                # debug draw
                distImg = cv2.drawChessboardCorners(img[1], self.patternSize, corners, ret)
                cv2.imwrite( str(self.imagePath) + "/dist/dist_" + str(img[0]).replace( str(self.imagePath) + "\\", ""), distImg)
                count += 1
            else:
                os.remove(str(img[0]))
                self.log.debug("don't detected corner")

        self.log.debug("detected image len is..." + str(count))

        if(len(imgPoints) > 0):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, self.imageSize, None, None)
            np.savetxt("img_20231005T184557_mtx.csv", mtx, delimiter=",", fmt="%0.14f") # カメラ行列
            np.savetxt("img_20231005T184557_dist.csv", dist, delimiter=",", fmt="%0.14f") # 歪みパラメータ
            # 計算結果を表示
            print("RMS:", ret)
            print("mtx:", mtx)
            print("dist:", dist)

            # 再投影誤差による評価
            mean_error = 0
            for i in range(len(objPoints)):
                image_points2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgPoints[i], image_points2, cv2.NORM_L2) / len(image_points2)
                mean_error += error
            print ("total error: ", mean_error/len(objPoints)) # 0に近いほど良い            

            return True
        else:
            self.log.debug("all image don't exist corner")
            return False

if __name__ ==  "__main__":
    path = "Image" # チェスボード画像ファイルが格納されているフォルダ
# チェスボードの交点
    rows = 10
    cols = 7

    size = 2.4 # チェスボードの辺の長さ
    calb = Calbration(path, cols=cols, rows=rows, squareSize=size)
    calb.calbration()
