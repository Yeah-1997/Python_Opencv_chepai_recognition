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
import net 
import net_clean


#========================================================
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
#=======================初始化=============================
print('加载SVM模型......\n')
svm_model = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/SVM_Python_数字识别/svm_model')
print('加载神经网络......\n')
# Hw = joblib.load('Hw')
# Ow = joblib.load('Ow')
Hw = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/jcs/Hw_0.3_20000')
Ow = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/jcs/Ow_0.3_20000')

# Hw = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/mes/Hw_0.3_20000')
# Ow = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/mes/Ow_0.3_20000')
# Hw = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/神经网络/w_jcs_Pra_[20, 12]_Acc1.000_0.989_sigmoid')
lossmode = 'j'#j:jcs m:mes

print('加载pca模型......\n')
pca = joblib.load('G:/我的地盘/毕设用/AA毕设\'s/SVM_Python_数字识别/pca_model')
print('加载完成\n')
clas=np.array(['A','B','0','1','2','3','4','5','6','7','8','9']) 
filename = 'G:/我的地盘/毕设用/AA毕设\'s/模糊矿车图片/'#车牌图片所在的位置
image_filenames = [os.path.join(filename, x) for x in os.listdir(filename) if prf.is_image_file(x)]  #得到每个图片路径

wrongnum = 0

for index,picadd in  enumerate(image_filenames):   #遍历每类的图片
    print('====================================================')
    realnum = (os.path.split(picadd)[1]).split('.')[0] #真实的车牌号
    print(str(index+1),'/',str(len(image_filenames)),'......')
    t = time()
#======================图像恢复===============================
    img = cv2.imdecode(np.fromfile(picadd,dtype=np.uint8),-1)#读取
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    print('图像读取用时：%f Seconds' % (time()-t))
    t = time()
    dis = prf.get_pra(imgGray)#模糊程度
    print('获取模糊参数：%f Seconds' % (time()-t))
    t = time()
    # print('模糊程度：\t ',dis)
    psf = prf.get_motion_dsf(imgGray.shape,0,int(dis))#获取点扩散函数
    print('产生PSF函数 ：%f Seconds' % (time()-t))
    t = time()
    imgRe = prf.wiener1(img,psf,channel=3) 
    print('维纳滤波用时：%f Seconds' % (time()-t))
    t = time()
#=====================空间转换================================
    imgHSV =cv2.cvtColor(imgRe,cv2.COLOR_BGR2HSV)
    # prf.cvshow(u'recovered image',imgRe)
    vch = imgHSV[:,:,2]#提取V通道
    print('空间转换时间：%f Seconds' % (time()-t))
    t = time()
#=====================车牌区域提取========================================
    ret,erzhi = cv2.threshold(vch,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # prf.cvshow(u'binerary image',erzhi)
    kernel = np.ones((5,5),np.uint8) 
    closing = cv2.morphologyEx(erzhi, cv2.MORPH_CLOSE, kernel)
    # prf.cvshow(u'closing ',closing)
    binary,cnts, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #轮廓选择
    # drawimg = imgRe.copy()
    # res = cv2.drawContours(drawimg, cnts, -1, (0, 0, 255), 2)
    # prf.cvshow('res',res)
    realcnt = np.array([])
    for i,x in enumerate(cnts):
        area = cv2.contourArea(x)
        if area>15000 and area<30000:
            realcnt = x
            break
    (x,y,w,h) = cv2.boundingRect(realcnt)
    chepai = imgRe[y:y+h,x:x+w,:]
    print('车牌定位时间：%f Seconds' % (time()-t))
    t = time()
#=============================车牌字符分割==================================
    # prf.cvshow(u'roi region',chepai)
    cpGray = cv2.cvtColor(chepai,cv2.COLOR_BGR2GRAY)
    mecp = cv2.medianBlur(cpGray,5)
    ret,bmcp = cv2.threshold(mecp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((30,12),np.uint8) 
    cbmcp = cv2.morphologyEx(bmcp, cv2.MORPH_CLOSE, kernel)
    # prf.cvshow(u'binerary region',cbmcp)
    binary,numcnts, hierarchy = cv2.findContours(cbmcp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #筛选下再 防止有噪音点
    numcandi = []
    for i,x in enumerate(numcnts):
        area = cv2.contourArea(x)
        if area>150 :
            numcandi.append(x) 
    numcandi = prf.sort_contours(numcandi, method="left-to-right")[0] #排序，从左到右，从上到下
    print('字符分割时间：%f Seconds' % (time()-t))
    t = time()
#============================识别===========================================
    plt.figure(figsize =(15,10) ,facecolor='w')
    result = []
    for i,c in enumerate(numcandi):
        (x, y, w, h) = cv2.boundingRect(c)	 #定位出数字坐标
        roi = mecp[y-2:y+h,x-2:x+w]
        roi = cv2.resize(roi,(30,30),interpolation=cv2.INTER_LINEAR)  
        x1 = roi.reshape(1,-1)

        # # y =  svm_model.predict(pca.transform(x1))#主成分分析后的向量送进去预测使用SVM
        if lossmode == 'j':
            y = net.netjcs_predict(pca.transform(x1),Hw,Ow)#使用损失函数是交叉熵的神经网络
        else:
            y = net.net_predict(pca.transform(x1),Hw,Ow)#使用神经网络
        # y = net.net_pro_jiaocha_4_elu_predict(pca.transform(x1),Hw)
        # y = net_clean.net_pro_jiaocha_predict(pca.transform(x1),Hw,func= 'sigmoid',loss = 'jcs')
        result.append(clas[y][0])
        #y =  svm_model.predict(x1)
        # plt.subplot(3,3,i+3)
        # plt.title(u'分为：%s' % clas[y][0])
        # plt.imshow(roi,cmap='gray') 
    
    result = "".join(result)
    output = ''
    for i,x in enumerate(result):
        if result[i] != realnum[i]:
            output += realnum[i]+'分为了' +result[i]+'\n'
            wrongnum += 1

    # plt.subplot(3,3,1)
    # plt.title(u'原图' )
    # plt.imshow(imgGray,cmap='gray')       
    # plt.subplot(3,3,2)
    # plt.title(u'复原图' )
    # plt.imshow(imgRe,cmap='gray')


    # plt.suptitle('识别结果:'+result+'\n'+'真实值:'+realnum+'\n'+output)  

    # plt.tight_layout(1.5)
    # plt.subplots_adjust(top=0.8)
    print('识别画图时间：%f Seconds' % (time()-t))
    print('识别结果:'+result+'\n'+'真实值:'+realnum+'\n'+output)
    # plt.show()
print('=====================================================')
print('共识别了%d张车牌\n分类了%d个字符\n字符分类正确率:%.2f%%'%(index+1,(index+1)*5,(1-(wrongnum/((index+1)*5)))*100))
