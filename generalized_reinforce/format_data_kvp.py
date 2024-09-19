import os 
import json

gts = os.listdir('test/gts')

reformatted = []
for i in gts:
    with open(os.path.join('test/gts', i), 'r') as file:

        obj = json.load(file)
        for item in obj['kvps_list']:
            if item['type'] == 'kvp' and item['key']['text'] != "" and item['value']['text'] != "":
                cur = {
                    'image': os.path.join('data/KVP10k_data/test/images/', i.split('.')[0] + '.png'),
                    'key': item['key'],
                    'value': item['value']
                }
                reformatted.append(cur)
with open('test.json', 'w') as file:
    json.dump(reformatted, file, indent=4)
