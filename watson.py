'''
Read data from csv file
TO BE NOTED : REARRANGEMENT OF COLUMNS IN mbti_1.csv

1. Invoke watson API (pip3 install -I watson-developer-cloud : create account and obtain credentials)
2. Get personality profile for each person's tweets (Big Five)
3. Map Big Five to MBTI
4. Accuracy of API based on exisitng classification

'''


from __future__ import print_function
import json
from os.path import join, dirname
from watson_developer_cloud import PersonalityInsightsV3
import csv

filename = "mbti_1.csv"
fields = []
rows = []
	 	
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	fields = next(csvreader)
	for row in csvreader:
		rows.append(row)

personality_insights = PersonalityInsightsV3(
	version = '2017-10-13',
	username = 'd0b4fa84-a7ed-40fa-8470-0411694fd337',
	password = 'TGFjF6k3eWrV')
	
classes = []
data = {}
for i in range(len(rows)):
	rows[i][0] = (" ").join([j for j in rows[i][0].split("|||")])
	classes.append(rows[i][1])
classes = list(set(classes))

for c in classes:
	data[c] = []
for i in range(len(rows)):
	data[rows[i][1]].append(rows[i][0])
#print(rows[:5])

accuracy = 0.0
for row in rows:
	profile_json = row[0]
	profile = personality_insights.profile(profile_json[1:len(profile_json)-1], content_type='text/plain;charset=utf-8',raw_scores=True,consumption_preferences=True)
	#print(row[1])
	predict = ''
	personality_prof = {}
	for d in profile['personality']:
		#print(d['name'],' ',d['percentile'],' ',d['raw_score'])
		personality_prof[d['name']] = float(d['raw_score'])
	if personality_prof['Extraversion'] >= 0.5:
		predict += 'E'
	else:
		predict += 'I'
	if personality_prof['Openness'] >= 0.5:
		predict += 'N'
	else:
		predict += 'S'
	if personality_prof['Agreeableness'] >= 0.5:
		predict += 'F'
	else:
		predict += 'T'
	if personality_prof['Conscientiousness'] >= 0.5:
		predict += 'J'
	else:
		predict += 'P'
	#print(predict)
	if row[1] == predict:
		accuracy += 1
print(float(accuracy/len(rows))*100)
		
				
