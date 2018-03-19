
from os.path import join, dirname
import csv
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
print(lmtzr.lemmatize('anticipation'))

#filename = "mbti_1.csv"
filename = "cleaned_mbti.csv"
fields = []
rows = []
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprize', 'trust']
    
with open(filename, 'r', encoding='mac_roman', newline='') as csvfile:
  csvreader = csv.reader(csvfile)
  fields = next(csvreader)
  for row in csvreader:
    rows.append(row)

nrc_lexicon = open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt').readlines()
emolex = {}
#print(nrc_lexicon[0])
for line in nrc_lexicon:
  vector = line.strip().split()
  #print(vector)
  if lmtzr.lemmatize(vector[0]) in emolex:
    emolex[lmtzr.lemmatize(vector[0])].append(vector[2])
  else:
    emolex[lmtzr.lemmatize(vector[0])] = [vector[2]]
#print(emolex)
#print(rows[:1])
#mbti
classes = []
data = {}
for i in range(len(rows)):
  classes.append(rows[i][0])
classes = list(set(classes))

tweets_vector = []
for row in rows:
  words = row[1].split(' ')
  tweet_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for word in words:
    vec = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    if word in emolex:
      vec = emolex[word]
    for i in range(len(emotions)):
      tweet_vec[i] += int(vec[i])
  #print(tweet_vec)
  tweets_vector.append(tweet_vec)
emo_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for tweet in tweets_vector:
  max_i = 0
  for i in range(len(tweet)):
    if i != 5 and i != 6:
      if tweet[i] > tweet[max_i]:
        max_i = i
  emo_vector[max_i] += 1
#print(emo_vector)
with open('emolex2.csv','w') as file:
  for i in range(len(rows)):
    file.write(rows[i][0]+",")
    for j in range(len(tweets_vector[i])):
      if j != 5 and j != 6 and j != 2 and j != 8:
        if j == len(tweets_vector[i])-1:
          file.write(str(tweets_vector[i][j]))
        else:
          file.write(str(tweets_vector[i][j])+",")
    file.write('\n')
#big-five
'''
ids = []
data = {}
for i in range(len(rows)):
  ids.append(rows[i][0])
ids = list(set(ids))


for i in range(len(rows)):
  if (rows[i][2],rows[i][3],rows[i][4],rows[i][5],rows[i][6]) in data:
    data[(rows[i][2],rows[i][3],rows[i][4],rows[i][5],rows[i][6])].append(rows[i][1])
  else:
    data[(rows[i][2],rows[i][3],rows[i][4],rows[i][5],rows[i][6])] = [rows[i][1]]
'''

'''
tweets_vector = {}
for tup in data:
  tweets_vector[tup] = []
  for tweet in data[tup]:
    words = word_tokenize(tweet)
    tweet_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for word in words:
      word = lmtzr.lemmatize(word)
      vec = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
      if word in emolex:
        vec = emolex[word]
      for i in range(len(emotions)):
        tweet_vec[i] += int(vec[i])
    tweets_vector[tup].append(tweet_vec)
#print(tweets_vector)

emo_vec = {}
for tup in tweets_vector:
  print(tup)
  emo_vec[tup] = {'anger' : 0, 'anticipation' : 0, 'disgust' : 0, 'fear' : 0, 'joy' : 0, 'negative' : 0, 'positive' : 0, 'sadness' : 0, 'surprize' : 0, 'trust' : 0}

  for tweet in tweets_vector[tup]:
    max_emotion = 0
    for i in range(len(emotions)):
      #print(emotions[i]+' : '+str(tweet[i]))
      if tweet[i] > tweet[max_emotion]:
        max_emotion = i
    #print(emotions[max_emotion])
    emo_vec[tup][emotions[max_emotion]] += 1
  print(sorted(emo_vec[tup].items(), key=lambda t: t[1], reverse = True))
'''   




'''
tweets_vector = {}
for row in rows:
  tweets = row[0].split("|||")
  tweet_vector = {}
  if row[1] not in tweets_vector:
    tweets_vector[row[1]] = []
  temp = [0,0,0,0,0,0,0,0,0,0]
  for tweet in tweets:
    words = word_tokenize(tweet)
    #print(words)
    #print()
    tweet_vector[tweet] = [0,0,0,0,0,0,0,0,0,0]
    words_vector = {}
    for word in words:
      words_vector[word] = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
      if word in emolex:
        words_vector[word] = emolex[word]
      for i in range(len(emotions)):
        tweet_vector[tweet][i] += int(words_vector[word][i])
    for i in range(len(emotions)):
      temp[i] += tweet_vector[tweet][i]
  tweets_vector[row[1]].append(temp)

emotional_lists = {}
for classification in tweets_vector:
  if classification not in emotional_lists:
    emotional_lists[classification] = []
  for emotion_list in tweets_vector[classification]:
    #print(emotion_list)
    d = {v: k for v, k in enumerate(emotion_list)}
    #print(d)
    sorted_vector = sorted(d.items(), key=lambda t: t[1], reverse = True)
    l = []
    for t in sorted_vector:
      l.append({emotions[t[0]]:t[1]})
    emotional_lists[classification].append(l)
#print(emotional_lists)
emotions_vector = {}
for classification in emotional_lists:
  #print(classification)
  #print(str(len(emotional_lists[classification])))
  emotions_vector[classification] = {}
  for tweets in emotional_lists[classification]:
    for d in tweets[0:3]:
      for k in d:
        if k in emotions_vector[classification]:
          emotions_vector[classification][k] += 1
        else:
          emotions_vector[classification][k] = 1
print(emotions_vector)
'''