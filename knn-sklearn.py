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

    #1      2      3      4      5      6      7      8      9      10     13     17     20     25     30     40     50     60     70     80     90     100
    #0.9691 0.9627 0.9705 0.9682 0.9688 0.9677 0.9694 0.9670 0.9659 0.9665 0.9653 0.9630 0.9625 0.9609 0.9596 0.9560 0.9534 0.9517 0.9487 0.9468 0.9452 0.9440
    #781.9  776.8  779.3  782.9  777.4  775.9  777.7  790.5  778.0  776.4  780.1  778.0  778.0  779.3  779.4  777.5  778.7  780.6  779.1  781.4  780.8  779.4