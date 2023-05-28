import joblib
import struct
import numpy as np
import os
from collections import Counter
import datetime

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


X_test, y_test = load_mnist('C:\\Users\\16907\\Desktop', kind='t10k')
print("测试集数据",Counter(y_test))
model1 = joblib.load("my_model_rbf.m")
predicted = [0 for _ in range(10)] #
correct = [0 for _ in range(10)]
st=datetime.datetime.now()
for i in range(X_test.shape[0]):
    a = model1.predict(X_test[i].reshape(1,784)/255)[0] #得到预测数据
    predicted[a] += 1
    if a == y_test[i]:
        correct[a] += 1
et = datetime.datetime.now()
print("运行时间 %s."%(et - st))
print("预测数据",predicted)
print("正确预测数据",correct)
preception_ratio = [0 for _ in range(10)]
for i in range(10):
    preception_ratio[i] = round(correct[i]*1.0/predicted[i],3)
print("查准率：",preception_ratio)
recall_ratio = [0. for _ in range(10)]
for i in range(10):
    recall_ratio[i] = round(correct[i]/Counter(y_test)[i],3)
print("召回率",recall_ratio)
F1 = [0 for _ in range(10)]
for i in range(10):
  F1[i] = round(2*preception_ratio[i]*recall_ratio[i]/(preception_ratio[i]+recall_ratio[i]),3)
print("F1值：",F1)



