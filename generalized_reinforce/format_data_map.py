import os
import json

input_json = "/home/raychai/MapQA_S/questions/test-QA.json"
image_dir = "/home/raychai/MapQA_S/images"
output_data = []

with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in data:
    map_id = entry["map_id"]
    question = entry["question"]
    answer = entry.get("answer", [])
    
    if isinstance(answer, list):
        answer_text = ", ".join(answer)
    else:
        answer_text = str(answer)
    
    image_path = os.path.join(image_dir, map_id)
    
    prompt = f"Given <image>, {question} What are the answers?"
    
    formatted_entry = {
        "image": image_path,
        "key": {
            "text": prompt
        },
        "value": {
            "text": answer_text
        }
    }
    
    output_data.append(formatted_entry)

output_file = "formatted_output.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Data has been formatted and saved to {output_file}")
