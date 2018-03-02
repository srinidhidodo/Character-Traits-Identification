import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import svm

ave_accuracy = 0.0
for i in range(10):
  print("Run ",(i+1))
  print()
  data = pd.read_csv("cleaned_mbti.csv",names=["MBTI_type","text"])
  X_temp = data["text"]
  #print(X_temp)
  #
  x_dict = X_temp.to_dict()
  Y = data["MBTI_type"]
  '''data['label_num'] = data.MBTI_type.map({'E':0, 'I':1})
  temp = data.label_num
  print(temp)'''
  '''temp = data.MBTI_type.map({'E':0, 'I':1})
  print(temp)'''

  l_temp = []
  for i in Y:
     l_temp.append(i[3])
  y_temp = pd.Series(l_temp)
  values = list(set(y_temp.tolist()))
  c = 0
  d = {}
  for i in values:
    d[i] = c
    c += 1
  temp = y_temp.map(d)
  y_temp = temp
  print(y_temp.value_counts())
  y_dict = y_temp.to_dict()
  #print(x_dict)

  
  l1 = []
  l2 = []
  l1_val = []
  l2_val = []
  for i in y_dict:
    if y_dict[i] == d[values[0]]:
      l1.append(d[values[0]])
      l1_val.append(x_dict[i])
    else:
      l2.append(d[values[1]])
      l2_val.append(x_dict[i])
  #n = int(random.randint(1000,1996))
  #n = int(random.randint(1000,1194))
  #n = int(random.randint(3000,3975))
  n = int(random.randint(3000,3429))
  l1_temp = []
  l2_temp = []
  l1_val_temp = []
  l2_val_temp = []
  #print(len(l1),' ',len(l1_val))
  #print(len(l2),' ',len(l2_val))
  xi_l = random.sample(range(0, len(l1)), n)
  for i in xi_l:
    xi = i
    l1_temp.append(l1[xi])
    l1_val_temp.append(l1_val[xi])
  
  xi_l = random.sample(range(0, len(l2)), n)

  for i in xi_l:
    xi = i
    #print(len(l2))
    #print(xi)
    l2_temp.append(l2[xi])
    l2_val_temp.append(l2_val[xi])
  l = l1_temp + l2_temp
  x = l1_val_temp + l2_val_temp


  X = pd.Series(x)


  y = pd.Series(l)
  #X = X_temp
  #y = y_temp
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
  vect = CountVectorizer()
  X_train_dtm = vect.fit_transform(X_train)
  X_test_dtm = vect.transform(X_test)
  model = svm.SVC(kernel='linear') 
  print(model.fit(X_train_dtm, y_train))
  print()
  print(model.score(X_train_dtm, y_train))
  predicted= model.predict(X_test_dtm)
  print("Accuracy = ",metrics.accuracy_score(y_test, predicted))
  ave_accuracy += metrics.accuracy_score(y_test, predicted)
  print()
  print()
ave_accuracy /= 10
print("Average accuracy = ",ave_accuracy)

  