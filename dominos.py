import requests as req
import json
import pandas as pd
 
headers = {'User-Agent': 'Mozilla/4.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
my_r = req.get('https://dominospizza.ru/find_dominos?st=del', headers=headers)
 
s = my_r.text
s = s[s.find('"geo":') + len('"geo":'):]
brackets = 1
r = 0
for i in range(1, len(s)):
    if brackets == 0:
        r = i
        break
    if s[i] == '{':
        brackets += 1
    elif s[i] == '}':
        brackets -= 1
j = json.loads(s[:r])
data = []
for i in j['cities']:
    data.append([i['longitude'], i['lattitude']])
data = pd.DataFrame(data, columns=['lon', 'lat'])
data.to_csv('dominos.csv')
