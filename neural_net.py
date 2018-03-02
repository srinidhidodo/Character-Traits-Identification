import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import random

ave_accuracy = 0.0
for j in range(10):
  print('Run : ',j)
  data = pd.read_csv("mbti_big5scores.csv",names=["E","O","A","C","N","MBTI_type"])
  #print(data.head)
  #print(data[:,5])
  X_temp = data.drop("MBTI_type",axis=1).astype(float)
  #print(X_temp)
  #
  x_dict = X_temp.to_dict()
  Y = data["MBTI_type"]

  l_temp = []
  for i in Y:
     l_temp.append(i[0])
  y_temp = pd.Series(l_temp)
  y_dict = y_temp.to_dict()
  #print(y_dict)

  values = list(set(y_temp.tolist()))
  l1 = []
  l2 = []
  l1_val = []
  l2_val = []
  for i in y_dict:
    if y_dict[i] == values[0]:
      l1.append(values[0])
      l1_val.append([x_dict['E'][i],x_dict['O'][i],x_dict['A'][i],x_dict['C'][i],x_dict['N'][i]])
    else:
      l2.append(values[1])
      l2_val.append([x_dict['E'][i],x_dict['O'][i],x_dict['A'][i],x_dict['C'][i],x_dict['N'][i]])
  n = int(random.randint(1000,1996))
  #n = int(random.randint(1000,1194))
  #n = int(random.randint(3000,3975))
  #n = int(random.randint(3000,3429))
  l2_temp = []
  l1_temp = []
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

  E = []
  O = []
  A = []
  C = []
  N = []
  for i in x:
    E.append(i[0])
    O.append(i[1])
    A.append(i[2])
    C.append(i[3])
    N.append(i[4])

  X = pd.DataFrame(
      {'E': E,
       'O': O,
       'A': A,
       'C': C,
       'N': N
      })
  #print(X)

  y = pd.Series(l)
  #X = pd.Data
  #print(X.size)
  #print(y.size)

  X_train, X_test, y_train, y_test = train_test_split(X, y)
  scaler = StandardScaler()
  #print(X_train)
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  mlp = MLPClassifier(hidden_layer_sizes=(200,50,16,),max_iter=100,verbose=False,solver='adam')
  print(mlp.fit(X_train,y_train))

  predictions = mlp.predict(X_test)
  print(confusion_matrix(y_test,predictions))
  print(classification_report(y_test,predictions))
  print("Accuracy = ",accuracy_score(y_test,predictions))
  ave_accuracy += accuracy_score(y_test,predictions)
ave_accuracy /= 10
print("Average accuracy = ",ave_accuracy)