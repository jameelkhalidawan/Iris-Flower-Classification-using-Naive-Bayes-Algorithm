#!/usr/bin/env python
# coding: utf-8

# In[45]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X = iris.data
y = iris.target
print("X_train")
print (len(X_train))
print(X_train)
print("\n\ny_train")
print (len(y_train))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()

gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("\n\n\nAccuracy:", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




