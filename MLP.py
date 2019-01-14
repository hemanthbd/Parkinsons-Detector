from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import svm
import scipy.io

mat1 = scipy.io.loadmat('train_input_H.mat')
mat2 = scipy.io.loadmat('train_input_PD.mat')
healthy = mat1['train_input_H']
pd = mat2['train_input_PD']
'''
print(pd.shape)
#print(pd[1,:400])
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(pd)

X_new = pca.transform(pd)
print(np.abs(pca.components_[0]).argsort()[::-1][:100])
print(np.abs(pca.components_[1]).argsort()[::-1][:100])
'''



data = np.zeros((100,680398));
data[:50,:] = healthy[:50,:]
data[50:100,:] = pd[:50,:]
train_labels = np.zeros((100,1))
train_labels[50:100,:]= 1

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(data, train_labels, test_size = 0.20, shuffle=True) 


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

from sklearn.neural_network import MLPClassifier  
#mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp = svm.SVC(gamma='scale')
mlp.fit(X_train, y_train.ravel())  

predictions = mlp.predict(X_test) 

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions)) 


X_test = np.zeros((20,156814));
X_test[:10,:] = healthy[40:50,:]
X_test[10:20,:] = pd[40:50,:]

train_labels = np.zeros((80,1))
train_labels[40:80,:]= 1

test_labels = np.zeros((20,1))
test_labels[10:20,:]= 1

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_train.ravel(), train_labels.ravel())

predicted = clf.predict(train_labels.ravel()) 

print(predicted)
