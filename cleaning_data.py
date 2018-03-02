import re
import csv
dataset = []
new_data = []
def repl(matchObj):
   char = matchObj.group(1)
   return "%s%s" % (char, char)
with open('original_mbti_bigfive.csv', 'r', encoding='mac_roman', newline='') as file:
  lines = csv.reader(file)
  dataset=list(lines)
#print(dataset[:5])
for row in dataset:
  text = re.sub(r"http\S+", "", row[1], flags=re.MULTILINE)
  text = (" ").join(text.split("|||"))
  new_text = ''
  if text != '' and text[0].isalnum():
    new_text += text[0]

  for i in range(1,len(text)-1):
    if text[i-1].isalpha() and text[i].isalpha() and text[i+1].isalpha():
      new_text += text[i]
    elif not text[i-1].isalpha() and text[i].isalpha() and text[i+1].isalpha():
      new_text += (' '+text[i])
    elif text[i-1].isalpha() and text[i].isalpha() and not text[i+1].isalpha():
      new_text += (text[i]+' ')
    elif not text[i-1].isalpha() and text[i].isalpha() and not text[i+1].isalpha():
      new_text += (' '+text[i]+' ')
    if text[i] == ' ':
      new_text += ' '
  new_text = re.sub(' +',' ',new_text)
  pattern = re.compile(r"(\w)\1+")
  new_text = pattern.sub(repl, new_text)
  new_text = new_text.lower()
  #print(new_text)
  s = row[0]+','+new_text
  new_data.append(s)
  
with open('cleaned_mbti.csv', 'w') as f:
  for line in new_data:
    f.write(line+"\n")