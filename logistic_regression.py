from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
#train a logistic regression classifier
# print(list(iris.keys()))
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])
# print(iris['data'].shape)  
x = iris["data"][:,3:]
y=  (iris["target"]==2).astype(np.int64)
print(y)
print(x)
clf=LogisticRegression()
clf.fit(x,y)
example= clf.predict(([[2.6]]))
print(example)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"g-",label='verginica')
plt.show()

