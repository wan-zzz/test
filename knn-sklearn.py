import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def loadData(fileName):
    print('start read file')
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    return dataArr, labelArr

if __name__ == "__main__":
    test_num=10000
    x_train,y_train = loadData('D:/大二/机器学习/Mnist/mnist_train.csv')
    x_test,y_test = loadData('D:/大二/机器学习/Mnist/mnist_test.csv')
    start = time.time()
    x_test1 = np.array(x_test)
    x_test2 = x_test1[0:test_num]
    y_test2=y_test[0:test_num]

    knn = KNeighborsClassifier(3)  # 引入训练方法
    knn.fit(x_train,y_train)  # 进行填充测试数据进行训练
    y_pre=knn.predict(x_test2)

    true_num=0
    for i in range(test_num):
        if y_pre[i]==y_test[i]:
            true_num+=1
    print(true_num / test_num)
    end = time.time()
    print('time span:', end - start)

    mat = confusion_matrix(y_test2, y_pre)
    ax=sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value K=3')
    plt.show()

 
