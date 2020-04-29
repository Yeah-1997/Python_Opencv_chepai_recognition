#========================================================
import numpy as np
import cv2
import os
import matplotlib.colors
import matplotlib.pyplot as plt
import math
from time import time 
from numpy import fft
from sklearn.decomposition import PCA
from functools import reduce
from matplotlib.pyplot import MultipleLocator
from sklearn import svm
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import profunc as prf

filename = 'G:/我的地盘/毕设用/AA毕设\'s/训练4.png'#车牌图片所在的位置
img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)#读取
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#先模糊在复原
psf = prf.get_motion_dsf(imggray.shape,0,40)
blur = prf.make_blurred(img,psf,0.01,channel=3)
bluegray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
# dis = prf.get_pra(blur[:,:,0])
# print(dis)
# psf1 = prf.get_motion_dsf(imggray.shape,0,40)
imgre = prf.wiener1(blur,psf,channel=3)
cpGray = cv2.cvtColor(imgre,cv2.COLOR_BGR2GRAY)
mecp = cv2.medianBlur(cpGray,5)
ret,bmcp = cv2.threshold(mecp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((30,12),np.uint8) 
cbmcp = cv2.morphologyEx(bmcp, cv2.MORPH_CLOSE, kernel)

binary,numcnts, hierarchy = cv2.findContours(cbmcp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #筛选下再 防止有噪音点
prf.cvshow('re',cbmcp)
numcandi = []
for i,x in enumerate(numcnts):
    area = cv2.contourArea(x)
    if area>150 :
        numcandi.append(x) 
numcandi = prf.sort_contours_complicated(numcandi, method="left-to-right")[0] #排序，从左到右，从上到下
drawimg = imgre.copy()
for i,x in enumerate(numcandi):
    res = cv2.drawContours(drawimg, x, -1, (0, 0, 255), 2)
    prf.cvshow('re',res)
print('开始保存')
for i,c in enumerate(numcandi):
    name = i%15+60
    fileadd = str(int(i/15))+'/'+str(name)+'.png'
    (x, y, w, h) = cv2.boundingRect(c)	 #定位出数字坐标
    roi = mecp[y-2:y+h,x-2:x+w]
    roi = cv2.resize(roi,(30,30),interpolation=cv2.INTER_LINEAR)  
    # prf.cvshow('re',roi)
    cv2.imwrite(fileadd,roi)
    # print(fileadd)


