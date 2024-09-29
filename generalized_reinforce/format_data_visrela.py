import os
import json

input_file = '/home/raychai/VisualRelationships/dataset/adobe/test.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

converted_data = []

for entry in data:
    img0 = os.path.join("/home/raychai/VisualRelationships/dataset/images", entry['img0'])
    img1 = os.path.join("/home/raychai/VisualRelationships/dataset/images", entry['img1'])
    
    sents = entry['sents']
    
    prompt = "What is the instruction given that changes from <image0> to <image1>?"
    
    if len(sents) > 0:
        instruction = sents[0]
        
        formatted_entry = {
            "image0": img0,
            "image1": img1,
            "key": {
                "text": prompt
            },
            "value": {
                "text": instruction
            }
        }
        
        converted_data.append(formatted_entry)

output_file = 'output_vlm_prompt.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print(f"Data has been saved to {output_file}")
