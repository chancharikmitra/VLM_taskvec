import json
import os

Code for turning data into json
with open('train.txt', 'r') as f:
    data = list(f)
    new_data = []
    for i in range(len(data)):
        cur_data = json.loads(data[i])
        for j in range(len(cur_data['annotations'])):

            new_data.append({'file_name' : os.path.join('data/wildreceipt', cur_data['file_name']),
                     'height' : cur_data['height'],
                     'width' : cur_data['width'], 
                     'annotation' : {'box' : cur_data['annotations'][j]['box'], 'text' : cur_data['annotations'][j]['text']},
                     'answer' : cur_data['annotations'][j]['label'] })
with open('train.json', 'w') as f:
    json.dump(new_data, f, indent=4)

# with open('class_list.txt', 'r') as f:
#     print(f.read())
