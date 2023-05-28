from __future__ import print_function
import matplotlib.pyplot as plt
import joblib
import cv2
import os
import struct
import numpy as np
from sklearn import svm
import datetime
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    # os.path.join()函数用于路径拼接文件路径
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


X_train, y_train = load_mnist('C:\\Users\\16907\\Desktop', kind='train')
b = np.linspace(255,255,784*60000).reshape(60000,784)
a = X_train[0].copy()
cv2.imwrite('1.jpg',a.reshape(28,28))
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('C:\\Users\\16907\\Desktop', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#3.训练svm分类器
temp = X_train/b #训练集特征缩放

st=datetime.datetime.now()
classifier = svm.SVC(C=1.0, kernel='poly',decision_function_shape='ovo')  # ovr:一对多策略
classifier.fit(temp, y_train.reshape(60000,1))  # ravel函数在降维时默认是行序优先
et = datetime.datetime.now()
print("Training spent : %s."%(et-st))

b = np.linspace(255,255,784*10000).reshape(10000,784)
temp1 = X_test/b #测试集特征缩放
# 4.计算svc分类器的准确率
print("训练集：", classifier.score(temp, y_train.reshape(60000,1)))
et1 = datetime.datetime.now()
print("Training dataset tseting spent %s."%(et1 - et))
print("测试集：", classifier.score(temp1, y_test.reshape(10000,1)))
et2 = datetime.datetime.now()
print("Testing dataset tseting spent %s."%(et2 - et1))
joblib.dump(classifier, "my_model_poly.m") #保存模型