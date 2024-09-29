import os
import json

input_file = "/home/raychai/spot-the-diff/data/annotations/test.json"
image_dir = "/home/raychai/resized_images"
output_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in data:
    img_id = entry["img_id"]
    sentences = entry["sentences"]
    
    image1 = os.path.join(image_dir, f"{img_id}.png")
    image2 = os.path.join(image_dir, f"{img_id}_2.png")
    
    prompt = f"Compare <image1> and <image2>. What differences do you observe?"
    
    differences = " ".join(sentences)
    
    formatted_entry = {
        "image1": image1,
        "image2": image2,
        "key": {
            "text": prompt
        },
        "value": {
            "text": differences
        }
    }
    
    output_data.append(formatted_entry)

output_file = "spotdiff_test.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Data has been formatted and saved to {output_file}")
