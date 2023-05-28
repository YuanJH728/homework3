# homework3
首先要把训练集和测试集解压成t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte，train-labels.idx1-ubyte，train-images.idx3-ubyte四个文件。
然后执行main.py文件，classifier = svm.SVC(C=1.0, kernel='poly',decision_function_shape='ovo')的kernal参数可取poly、linear和rbf，从而生成三个模型，并保存模型。
preception.py文件需要先模型加载，然后计算出查准率、召回率和F1值并输出。
