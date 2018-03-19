import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm



ave_accuracy = 0.0
for i in range(10):
  print("Run ",(i+1))
  print()
  data = pd.read_csv("cleaned_mbti.csv",names=["MBTI_type","text"])
  
  #data = pd.read_csv("mbti_big5scores.csv",names=["E","O","A","C","N","MBTI_type"])

  X_temp = data["text"]
  #X_temp = data.drop("MBTI_type",axis=1).astype(float)

  #print(X_temp)
  #
  x_dict = X_temp.to_dict()
  Y = data["MBTI_type"]
  '''data['label_num'] = data.MBTI_type.map({'E':0, 'I':1})
  temp = data.label_num
  print(temp)'''
  '''temp = data.MBTI_type.map({'E':0, 'I':1})
  print(temp)'''

  n = int(random.randint(1000,1194))
  s1_actual = []
  s1_pred = []
  df_d_actual = {}
  df_d_pred = {}
  x1_l = []
  x2_l = []
  index = [0,1,2,3]
  for ind in index:
    l_temp = []
    for i in Y:
       l_temp.append(i[ind])
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
    #print(y_dict)

    
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
  
    '''for i in y_dict:
      print(y_dict[i],' ',values[0])
      if y_dict[i] == d[values[0]]:

        l1.append(d[values[0]])
        l1_val.append([x_dict['E'][i],x_dict['O'][i],x_dict['A'][i],x_dict['C'][i],x_dict['N'][i]])
      else:
        l2.append(d[values[1]])
        l2_val.append([x_dict['E'][i],x_dict['O'][i],x_dict['A'][i],x_dict['C'][i],x_dict['N'][i]])'''
    
    #n = int(random.randint(1000,1194))
    #n = int(random.randint(1000,1194))
    #n = int(random.randint(3000,3975))
    #n = int(random.randint(3000,3429))
    l1_temp = []
    l2_temp = []
    l1_val_temp = []
    l2_val_temp = []
    #print(len(l1),' ',len(l1_val))
    #print(len(l2),' ',len(l2_val))
    #n = min(n,len(l1))
    #print(n,' ',len(l1))
    #print(n,' ',len(l2))
    if len(x1_l) == 0:
      x1_l = random.sample(range(0, 1194), n)
    
    for i in x1_l:
      xi = i
      l1_temp.append(l1[xi])
      l1_val_temp.append(l1_val[xi])
    #n = min(n,len(l2))
    if len(x2_l) == 0:
      x2_l = random.sample(range(0, 1194), n)
    
    for i in x2_l:
      xi = i
      #print(len(l2))
      #print(xi)
      l2_temp.append(l2[xi])
      l2_val_temp.append(l2_val[xi])
    l = l1_temp + l2_temp
    x = l1_val_temp + l2_val_temp


    X = pd.Series(x)
    '''E = []
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
        })'''

    y = pd.Series(l)
    #X = X_temp
    #y = y_temp
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    vect = CountVectorizer()
    #vect = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    X_train_dtm = vect.fit_transform(X_train)
    print(X_train_dtm.shape)
    X_test_dtm = vect.transform(X_test)
    print(X_test_dtm.shape)

    nb = hmm.GaussianHMM(n_components=2)
    #nb = MultinomialNB()
    #nb = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
    print(nb.fit(X_train_dtm.toarray(), y_train))
    y_pred_class = nb.predict(X_test_dtm.toarray())
    #print(X)

    print("Accuracy = ",metrics.accuracy_score(y_test, y_pred_class))
    #ave_accuracy += metrics.accuracy_score(y_test, y_pred_class)
    null_accuracy = y_test.value_counts().head(1) / len(y_test)
    print('Null accuracy:', null_accuracy)
    print(metrics.confusion_matrix(y_test, y_pred_class))
    #df_d_actual[ind] = y_test;
    #df_d_pred[ind] = y_pred_class;
    #print(y_test)
    if ind == 0:
      #s1_actual = y_test
      #s1_pred = pd.Series(y_pred_class)
      lt = y_test.tolist()

      for i in range(len(lt)):
        s1_actual.append(str(lt[i]))
        s1_pred.append(str(y_pred_class[i]))


    else:
      lt = y_test.tolist()
      for i in range(len(lt)):
        s1_actual[i] += str(lt[i])
        s1_pred[i] += str(y_pred_class[i])
      #s1_actual.append(y_test)
      #s1_pred.append(pd.Series(y_pred_class))
    #print(s1_actual)
    #print(y_pred_class)

    


    #print('message text for the false positives')

    #print(X_test[y_pred_class > y_test])

    # alternative less elegant but easier to understand
    # X_test[(y_pred_class==1) & (y_test==0)]
    #print()
    #print('message text for the false negatives')

    #print(X_test[y_pred_class < y_test])
    # alternative less elegant but easier to understand
    # X_test[(y_pred_class=0) & (y_test=1)]
    #print()
    #print('calculate predicted probabilities for X_test_dtm (poorly calibrated)')

    # Numpy Array with 2C
    # left Column: probability class 0
    # right C: probability class 1
    # we only need the right column 
    #y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
    #print(y_pred_prob)
    #print()
    #print('calculate AUC')
    #print(metrics.roc_auc_score(y_test, y_pred_prob))
    # Naive Bayes predicts very extreme probabilites, you should not take them at face value
    print()
    print()
    print()

  #print(type(df_d_actual[0]))
  #final_df_actual = pd.concat([df_d_actual[0],df_d_actual[1],df_d_actual[2],df_d_actual[3]])
  #final_df_pred = pd.concat([df_d_pred[0],df_d_pred[1],df_d_pred[2],df_d_pred[3]])
  print("Combined Accuracy : ",metrics.accuracy_score(pd.Series(s1_actual), pd.Series(s1_pred)))
  ave_accuracy += metrics.accuracy_score(pd.Series(s1_actual), pd.Series(s1_pred))
ave_accuracy /= 10
print("Average accuracy = ",ave_accuracy)

