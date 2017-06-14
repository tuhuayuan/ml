# pylint: disable=all

#%%
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
fig.add_subplot(221)
xx1, xx2 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.contourf(xx1, xx2, xx1**2 + xx2**2)

fig.add_subplot(222) 
xx1, xx2 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
z1 = xx1.copy()
z2 = xx2.copy()
z1[z1>=0] = 1
z1[z1<0] = -1
z2[z2>=0] = 2
z2[z2<0] = -2
plt.contourf(xx1, xx2, z1+z2+10)

fig.add_subplot(223) 
xx1, xx2 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.contourf(xx1, xx2, xx1+xx2)

fig.add_subplot(224) 
xx1, xx2 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.contourf(xx1, xx2, xx1+2*xx2)

#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
ax1.plot(randn(50).cumsum(), 'k--')
x = np.arange(0., 5., 0.2)
ax2.plot(x, np.sin(x), 'g--')
ax2.axis([0, np.pi * 2, -1.5, 1.5])
ax3.hist(randn(1000), bins=50, color='k')
ax4.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier,
                          test_idx=None, resolution=0.02):
    """显示decision regions"""

    # 初始化显示的标记和颜色
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
      np.arange(x1_min, x1_max, resolution),
      np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx1.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test set')                
#%%
"""训练感知器模型"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 分割出CV数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)

# 标准化参数
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Accuracy: %.2f ' % accuracy_score(y_test, y_pred))

# 合并测试和训练集
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105,150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()