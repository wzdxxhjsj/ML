# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division

import numpy as np  


from sklearn import tree as skt


from sklearn.model_selection import train_test_split
from sklearn import preprocessing  



data   = []  
labels = []  

with open("data.txt") as ifile:  
        for line in ifile:  
            tokens = line.strip().split('\t')  
            if tokens!=['']:
    #            data.append([float(tk) for tk in tokens[:]])  
                data.append([float(tk) for tk in tokens[:-1]])  
                labels.append(tokens[-1]) 
x = np.array(data) 
y = np.array(labels).astype(int) 
            


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1) 


''''' DecisionTree '''  




DecisionTree=skt.DecisionTreeClassifier()
DecisionTree.fit(x_train, y_train)

y_predict=DecisionTree.predict(x_test)

''''' DecisionTree '''




''''' accuracy_train '''

y_predict_train=DecisionTree.predict(x_train)

Y1=np.double(y_train)
Y2=np.double(y_predict_train)

Size=0
for k in range(0,Y1.size):  
    if(Y1[k]==Y2[k]):
        Size+=1
        
accuracy =   Size/Y1.size*100

print('accuracy_train',accuracy)






''''' accuracy_test '''


Y1=np.double(y_test)
Y2=np.double(y_predict)


#accuracy =   np.size(find(predictLabel == testLabel))/np.size(testLabel)*100
Size=0
for k in range(0,Y1.size):  
    if(Y1[k]==Y2[k]):
        Size+=1
        
accuracy =   Size/Y1.size*100

print('accuracy_test',accuracy)



''''''''


import matplotlib.pyplot as plt  


fig = plt.figure(figsize=plt.figaspect(0.5))


maxy=int(max(y_predict))
miny=int(min(y_predict))
K=maxy-miny+1
c=np.random.random([K,3])


ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('real')
ax.view_init(30, 60)

for k in range(0,Y1.size):    
    co=(c[np.mod(int(Y1[k]),4),0],c[np.mod(int(Y1[k]),4),1],c[np.mod(int(Y1[k]),4),2])
    ax.scatter(x_test[k,0],x_test[k,1],x_test[k,2],color=co)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('DecisionTree')
ax.view_init(30, 60)


for k in range(0,Y2.size):    
    co=(c[np.mod(int(Y2[k]),4),0],c[np.mod(int(Y2[k]),4),1],c[np.mod(int(Y2[k]),4),2])
    ax.scatter(x_test[k,0],x_test[k,1],x_test[k,2],color=co)


 