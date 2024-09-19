#### 
import json
import random
# from datasets import load_dataset
import PIL
from PIL import ImageDraw
from PIL import ImageFont
####

####Task Prompts
vizwiz_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. 
If the question can not be answered, respond unanswerable. """

okvqa_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. """

kvp_prompt = """
Given the image as well as the text and location of a certain key, extract the corresponding value.
"""

# Copied from wildreceipts/class_list.txt
wildreceipt_options_string = """
Ignore
Store_name_value
Store_name_key
Store_addr_value
Store_addr_key
Tel_value
Tel_key
Date_value
Date_key
Time_value
Time_key
Prod_item_value
Prod_item_key
Prod_quantity_value
Prod_quantity_key
Prod_price_value
Prod_price_key
Subtotal_value
Subtotal_key
Tax_value
Tax_key
Tips_value
Tips_key
Total_value
Total_key
Others"""
# wildreceipt_options_string = """
# 0 Ignore
# 1 Store_name_value
# 2 Store_name_key
# 3 Store_addr_value
# 4 Store_addr_key
# 5 Tel_value
# 6 Tel_key
# 7 Date_value
# 8 Date_key
# 9 Time_value
# 10 Time_key
# 11 Prod_item_value
# 12 Prod_item_key
# 13 Prod_quantity_value
# 14 Prod_quantity_key
# 15 Prod_price_value
# 16 Prod_price_key
# 17 Subtotal_value
# 18 Subtotal_key
# 19 Tax_value
# 20 Tax_key
# 21 Tips_value
# 22 Tips_key
# 23 Total_value
# 24 Total_key
# 25 Others"""

wildreceipt_prompt = """
Given the image, its dimensions, some text, and its precise bounding box location, classify the text as one of the 25 categories provided.
"""

wildreceipt_options = ['Ignore', 
'Store_name_value',
'Store_name_key',
'Store_addr_value',
'Store_addr_key',
'Tel_value',
'Tel_key',
'Date_value',
'Date_key',
'Time_value',
'Time_key',
'Prod_item_value',
'Prod_item_key',
'Prod_quantity_value',
'Prod_quantity_key',
'Prod_price_value',
'Prod_price_key',
'Subtotal_value',
'Subtotal_key',
'Tax_value',
'Tax_key',
'Tips_value',
'Tips_key',
'Total_value',
'Total_key',
'Others']

####

def open_data(dataset_name, path):

    jsonl_format_dataset = ["vizwiz", "okvqa", "ai2d", "hateful", "chart", "pope", "refcoco", "wino"]
    list_format_dataset = ["flower", "cub", "icon", "food", "dtd", "foci", "classify", "ocr", "mmmu", "text_task", "sst2", "xray"]


    if dataset_name == "flickr":
        dataset = json.load(json_file)["annotations"]
        return dataset

    if dataset_name == "path":
        if path == "train":
            path = path + "[:10%]"
        dataset = load_dataset("flaviagiammarino/path-vqa", split=path)
        dataset = list(dataset)
        formatted_dataset = []
        for item in dataset:
            if item["answer"] == "yes" or item["answer"] == "no":
                formatted_dataset.append(item)
        return formatted_dataset

    with open(path, 'r') as json_file:
        if dataset_name in jsonl_format_dataset:
            dataset = list(json_file)
        elif dataset_name in list_format_dataset:
            dataset = json.load(json_file)
        elif dataset_name == "info":
            dataset = json.load(json_file)["data"]
    return dataset




### Each format function should return (full_text, image_list, answer, question_id)
def get_format_func(cur_dataset):

    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "food":
        return format_food
    if cur_dataset == "ai2d":
        return format_ai2d
    if cur_dataset == "info":
        return format_info
    if cur_dataset == "icon":
        return format_icon
    if cur_dataset == "dtd":
        return format_dtd
    if cur_dataset == "foci":
        return format_foci
    if cur_dataset == "classify":
        return format_classify
    if cur_dataset == "hateful":
        return format_hateful
    if cur_dataset == "path":
        return format_path
    if cur_dataset == "ocr":
        return format_ocr
    if cur_dataset == "chart":
        return format_chart
    if cur_dataset == "pope":
        return format_pope
    if cur_dataset == "refcoco":
        return format_refcoco
    if cur_dataset == "flickr":
        return format_flickr
    if cur_dataset == "mmmu":
        return format_mmmu
    if cur_dataset == "text_task":
        return format_text_task
    if cur_dataset == "wino":
        return format_wino
    if cur_dataset == "sst2":
        return format_sst2
    if cur_dataset == "xray":
        return format_xray
    if cur_dataset == "kvp":
        return format_kvp
    if cur_dataset == "wildreceipt":
        return format_wildreceipt


####All return format will be in the form (Text, list of images, Answer, Question_id)
def format_vizwiz(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:

        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append("../" + sample["image"])

        full_text = vizwiz_prompt + few_shot_prompt + prompt.format(question)
    else:
        full_text = few_shot_prompt + prompt.format(question)

    image_list.append("../" + image)

    return full_text, image_list, answer, question_id

def format_kvp(all_data, cur_item=None, num_shot=0):
    # Should this be different for Qwen or should all models use the bounding box ref format.
    prompt = '{}<image> Key Text: {} Key Bounding Box: {} \n\nValue: '
    image_list = []

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]
    image, key_text, key_loc, value_text = cur_item['image'], cur_item['key']['text'], cur_item['key']['bbox'], cur_item['value']['text']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(kvp_prompt, sample['key']['text'], sample['key']['bbox']) + sample['value']['text'] #f" {str(sample['answer']) + ' ' + wildreceipt_options[sample['answer']]}  
            image_list.append("../" + sample["image"])
    # In the case of wildreceipts, the prompt should always be added to each one as above. Reevaluate later if needed.
    full_text = few_shot_prompt + prompt.format(kvp_prompt, key_text, key_loc)
    # if num_shot != 0:
    #     full_text = wildreceipt_prompt + few_shot_prompt + prompt.format(question)
    # else:
    #     full_text = few_shot_prompt + prompt.format(question)
    image_list.append("../" + image)

    # Question ID not needed for Wildreceipt
    question_id = 0
    target = f'{value_text}'
    return full_text, image_list, target, question_id
    
def format_wildreceipt(all_data, cur_item=None, num_shot=0):
    # Should this be different for Qwen or should all models use the bounding box ref format.
    prompt = '{}<image> Image Width: {} Image Height: {} Annotation: {} Categories: {} \n\nAnswer: '
    image_list = []

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]
    image, width, height, annotation, answer = cur_item['file_name'], cur_item['width'], cur_item['height'], cur_item['annotation'], cur_item['answer']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(wildreceipt_prompt, sample['width'], sample['height'], sample['annotation'], wildreceipt_options_string) + wildreceipt_options[sample['answer']] #f" {str(sample['answer']) + ' ' + wildreceipt_options[sample['answer']]}  
            image_list.append("../" + sample["file_name"])
    # In the case of wildreceipts, the prompt should always be added to each one as above. Reevaluate later if needed.
    full_text = few_shot_prompt + prompt.format(wildreceipt_prompt, width, height, annotation, wildreceipt_options_string)
    # if num_shot != 0:
    #     full_text = wildreceipt_prompt + few_shot_prompt + prompt.format(question)
    # else:
    #     full_text = few_shot_prompt + prompt.format(question)
    image_list.append("../" + image)

    # Question ID not needed for Wildreceipt
    question_id = 0
    target = f'{wildreceipt_options[answer]}'
    return full_text, image_list, target, question_id

# def format_okvqa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
#     prompt = '<image>{} Answer:'

#     image_list = []

#     if cur_item is None:
#         data = json.loads(random.sample(all_data, 1)[0])
#     else:
#         data = json.loads(cur_item)

#     image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

#     few_shot_prompt = ''
#     if num_shot > 0:
#         sampled_data = random.sample(all_data, num_shot)
#         for sample in sampled_data:
#             sample = json.loads(sample.strip())
#             few_shot_prompt += prompt.format(sample['question']) + f"{sample['answer']}."
#             image_list.append(sample["image"])

    
#     full_text = few_shot_prompt + prompt.format(question)
#     image_list.append(image)


#     return full_text, image_list, answer, question_id


###For llava and instruction inspections
def format_okvqa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{}'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:
        # sampled_data = random.sample(all_data, num_shot)
        # for sample in sampled_data:
        #     sample = json.loads(sample.strip())
        #     few_shot_prompt += prompt.format(sample['question']) + f"{sample['answer']}."
        #     image_list.append(sample["image"])

        full_text = few_shot_prompt + prompt.format(question) + "Answer the question with a single word."
    else:
        full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)


    return full_text, image_list, answer, question_id




def foci_helper(options, answer):
    cur_q = "Which of these flowers is shown in the image? Choices:\n"
    gt_letter = None
    random.shuffle(options)
    for item, letter in zip(options, ["A", "B", "C", "D"]):
        cur_q += f"{letter}. {item}\n"
        if item == answer:
            gt_letter = letter
    
    return cur_q, gt_letter


def format_foci(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{}Answer with the letter from the given choices directly. Answer:'

    image_list = []
    data = cur_item

    image, options, answer = data['image'], data['options'], data['groundtruth']

    question, letter_ans = foci_helper(options, answer)
    full_text = prompt.format(question)
    image_list.append("/datasets/flowers_2024-01-10_1812/flowers-102/" + image)

    return full_text, image_list, letter_ans, -1


def classify_helper(all_data, cur_class, rand_num):

    if rand_num:
        image = random.sample(all_data[cur_class], 1)[0]
        answer = "Yes"

        return image, answer
    else:
        neg_class = random.sample(list(all_data.keys()), 1)[0]
        while neg_class == cur_class:
           neg_class = random.sample(list(all_data.keys()), 1)[0]
        image = random.sample(all_data[neg_class], 1)[0]
        answer = "No"

    return image, answer


def format_classify(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    cur_class = model_helper.classifier_name
    prompt = f'<image>\nIs this a {cur_class}? Answer the question with Yes or No.'

    ###Brandon. For compressing prompt
    #prompt = f'<image>\nIs this a {cur_class}?'

    rand_num = random.randint(0,1)

    if cur_item is None:
        #image, answer = classify_helper(all_data, cur_class, rand_num)

        ##This is for datasetvector / Using the test set during Reinforce
        image, answer = random.sample(all_data, 1)[0]
    else:

        image, answer = cur_item

    image_list = []

    few_shot_prompt = ''
    if num_shot > 0:

        for _ in range(2):

            cur_image, cur_answer = classify_helper(all_data, cur_class, 1)
            image_list.append(cur_image)
            few_shot_prompt += prompt + f" {cur_answer}\n"


            cur_image, cur_answer = classify_helper(all_data, cur_class, 0)
            image_list.append(cur_image)
            few_shot_prompt += prompt + f" {cur_answer}\n"
        

    full_text = few_shot_prompt + prompt
    image_list.append(image)
    return full_text, image_list, answer, -1



# def format_classify(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
#     cur_class = model_helper.classifier_name
#     prompt = f'<|im_start|>user\n<image>\nIs this a {cur_class}?'
#     start_prompt = f'<image>\nIs this a {cur_class}?'

#     ###Brandon. For compressing prompt
#     #prompt = f'<image>\nIs this a {cur_class}?'

#     rand_num = random.randint(0,1)

#     if cur_item is None:
#         image, answer = classify_helper(all_data, cur_class, rand_num)
#     else:

#         image, answer = cur_item

#     image_list = []

#     few_shot_prompt = ''
#     if num_shot > 0:

#         for _ in range(2):

#             cur_image, cur_answer = classify_helper(all_data, cur_class, 1)
#             image_list.append(cur_image)

#             if _ == 0:
#                 few_shot_prompt += start_prompt + "<|im_end|>\n<|im_start|>assistant\n" + f"{cur_answer}<|im_end|>\n"
#             else:
#                 few_shot_prompt += prompt + "<|im_end|>\n<|im_start|>assistant\n" + f"{cur_answer}<|im_end|>\n"

#             cur_image, cur_answer = classify_helper(all_data, cur_class, 0)
#             image_list.append(cur_image)
#             few_shot_prompt += prompt + "<|im_end|>\n<|im_start|>assistant\n" + f"{cur_answer}<|im_end|>\n"

#         full_text = few_shot_prompt + prompt
#     else:
#         full_text = start_prompt

#     image_list.append(image)
#     return full_text, image_list, answer, -1



def sst2_helper(all_data, rand_num):
    if rand_num:
        image = random.sample(all_data["positive"], 1)[0]
        answer = "positive"

    else:
        image = random.sample(all_data["negative"], 1)[0]
        answer = "negative"
    return image, answer


def format_sst2(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = f'<image>\nIs the sentiment in the image positive or negative? Answer the question with a single phrase.'

    rand_num = random.randint(0,1)

    if cur_item is None:
        image, answer = sst2_helper(all_data, rand_num)
    else:
        image, answer = cur_item

    image_list = []

    few_shot_prompt = ''
    if num_shot > 0:

        for _ in range(2):

            cur_image, cur_answer = sst2_helper(all_data, 1)
            image_list.append(cur_image)
            few_shot_prompt += prompt + f" {cur_answer}\n"

            cur_image, cur_answer = sst2_helper(all_data, 0)
            image_list.append(cur_image)
            few_shot_prompt += prompt + f" {cur_answer}\n"

    full_text = few_shot_prompt + prompt
    image_list.append(image)
    return full_text, image_list, answer, -1


def xray_helper(all_data, cur_class, rand_num):

    if rand_num:
        image = random.sample(all_data[cur_class], 1)[0]
        answer = "Yes"
        return image, answer
    else:
        neg_class = random.sample(list(all_data.keys()), 1)[0]
        while neg_class == cur_class:
           neg_class = random.sample(list(all_data.keys()), 1)[0]
        image = random.sample(all_data[neg_class], 1)[0]
        answer = "No"
    return image, answer



# def format_xray(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
#     cur_class = model_helper.classifier_name
#     prompt = f'<image>\nIs this a {cur_class} x-ray? Answer the question with Yes or No.'

#     ###Brandon. For compressing prompt
#     #prompt = f'<image>\nIs this a {cur_class}?'

#     rand_num = random.randint(0,1)

#     if cur_item is None:
#         image, answer = xray_helper(all_data, cur_class, rand_num)
#     else:
#         image, answer = cur_item
#         if answer == cur_class:
#             answer = "Yes"
#         else:
#             answer = "No"

#     image_list = []

#     few_shot_prompt = ''
#     if num_shot > 0:

#         for _ in range(2):

#             cur_image, cur_answer = xray_helper(all_data, cur_class, 1)
#             image_list.append(cur_image)
#             few_shot_prompt += prompt + f" {cur_answer}\n"

#             cur_image, cur_answer = xray_helper(all_data, cur_class, 0)
#             image_list.append(cur_image)
#             few_shot_prompt += prompt + f" {cur_answer}\n"

#     full_text = few_shot_prompt + prompt
#     image_list.append(image)
#     return full_text, image_list, answer, -1


def format_xray(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    label_list = ["normal", "covid", "pneumonia"]
    random.shuffle(label_list)


    cur_class = random.sample(label_list, 1)[0]
    if cur_item is None:
        rand_num = random.randint(0,1)
        image, answer = xray_helper(all_data, cur_class, 1)
        answer = cur_class
    else:
        image, answer = cur_item 


    label1 = label_list[0]
    label2 = label_list[1]
    label3 = label_list[2]
    label_dict = {label1: "A", label2: "B", label3:"C"}


    # if num_shot==0:
    #     image_list = [image]
    #     example1 = ""
    #     example2 = "" 
    #     example3 = ""
    # else:
    image1, _ = xray_helper(all_data, label1, 1)
    image2, _ = xray_helper(all_data, label2, 1)
    image3, _ = xray_helper(all_data, label3, 1)


    example1 = f"<image>\nWhat is the type of this medical x-ray image? A.{label1} B.{label2} C.{label3}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
    example2 = f"<image>\nWhat is the type of this medical x-ray image? A.{label1} B.{label2} C.{label3}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
    example3 = f"<image>\nWhat is the type of this medical x-ray image? A.{label1} B.{label2} C.{label3}\nAnswer with the option's letter from the given choice directly. Answer: C\n"
    image_list = [image1, image2, image3, image]


    example4 = f"<image>\nWhat is the type of this medical x-ray image? A.{label1} B.{label2} C.{label3}\nAnswer with the option's letter from the given choice directly. Answer:"

    return example1 + example2 + example3 + example4, image_list, label_dict[answer], -1


def process_classify_val(val_data, model_helper):
    cur_class = model_helper.classifier_name
    processed_data = []
    neg_count_per_class = len(val_data[cur_class]) // 9
    for item in val_data[cur_class]:
        processed_data.append((item, "Yes"))

    
    for cur_key in val_data.keys():
        if cur_key == cur_class:
            continue
        sampled_data = random.sample(val_data[cur_key], neg_count_per_class)
        for temp in sampled_data:
            processed_data.append((temp, "No"))
    return processed_data




def format_flower(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        # pos_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        # neg_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return cur_query, [query], query_label, -1
        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        # pos_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        # neg_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return cur_query, [query], query_label, -1
        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_food(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]


    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of food in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of food in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of food in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of food in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of food in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of food in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_cub(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        # pos_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        # neg_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return cur_query, [query], query_label, -1
    else:
        # pos_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        # neg_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return cur_query, [query], query_label, -1
    

def format_dtd(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    
    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        # pos_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        # neg_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return cur_query, [query], query_label, -1
    else:
        # pos_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        # neg_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return cur_query, [query], query_label, -1


def format_ai2d(all_data, cur_item = None, num_shot=0, model_helper=None, split="train"):
    def parse_question(question_str, ans, image_path):
        return  f"<image>{question_str} Answer with the given option directly.", f"/home/chancharikm/taskvec/VLM_taskvec/{image_path}", ans

    image_list = []
    if cur_item is None:
        cur_item = json.loads(random.sample(all_data, 1)[0])
    else:
        cur_item = json.loads(cur_item)
    cur_prompt, cur_image, cur_ans = parse_question(cur_item["question"], cur_item["answer"], cur_item["image"])
    cur_id = cur_item["question_id"]

    few_shot_str = ''
    if num_shot > 0:
        samples = random.sample(all_data, num_shot)
        for sample in samples:
            sample = json.loads(sample)
            sample_prompt, sample_image, sample_ans = parse_question(sample["question"], sample["answer"], sample["image"])
            few_shot_str += sample_prompt + f" {sample_ans}"
            image_list.append(sample_image)
    image_list.append(cur_image)
    return few_shot_str + cur_prompt, image_list, cur_ans, cur_id


def format_info(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    # prompt = '<img>/home/zhaobin/Qwen-VL/task_vector/infovqa/data/{}</img>{} Answer: {}'
    # query_prompt = '<img>/home/zhaobin/Qwen-VL/task_vector/infovqa/data/{}</img>{} Answer:'


    prompt = "<image>{} Answer the question with a single word (or phrase). {}"
    image_list = []


    if cur_item is None:
        cur_query = random.sample(all_data, 1)[0]
        # while cur_query["evidence"][0] != "text":
        #     cur_query = random.sample(all_data, 1)[0]
    else:
        cur_query = cur_item


    few_shot_str = ''


    if num_shot > 0:
        samples = random.sample(all_data, num_shot)
        for sample in samples:

            # correct_type = False
            # while not correct_type:
            #     sample = random.sample(all_data, 1)[0]
            #     if sample["evidence"][0] == problem:
            #         correct_type = True
            #few_shot_str += prompt.format(sample["image_local_name"], sample["question"], sample["answers"][0])

            few_shot_str += prompt.format(sample["question"], sample["answers"][0])
            image_list.append(f"/home/zhaobin/Qwen-VL/task_vector/infovqa/data/{sample['image_local_name']}")


    image_list.append(f"/home/zhaobin/Qwen-VL/task_vector/infovqa/data/{cur_query['image_local_name']}")
    final_out = few_shot_str + f"<image>{cur_query['question']} Answer the question with a single word (or phrase)."
    return final_out, image_list, cur_query["answers"][0], cur_query["answers"]


# all_data:list of pid, cur_item: pid
def format_icon(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    img_path = "/home/zhaobin/Qwen-VL/task_vector/icon/iconqa_data/iconqa/{}/fill_in_blank/{}/image.png"

    prompt = '<image>{} Answer the question with a single word or phrase.'

    image_list = []

    if cur_item is None:
        cur_pid = random.sample(all_data, 1)[0]
        cur_data = model_helper.question_lookup[cur_pid]

    else:
        cur_pid = cur_item
        cur_data = model_helper.question_lookup[cur_item]

    cur_question, cur_img_path, cur_ans = cur_data["question"], img_path.format(split, cur_pid), cur_data['answer']
    
    few_shot_str = ''
    if num_shot > 0:
        samples = random.sample(all_data, num_shot)
        for sample in samples:

            sample_data = model_helper.question_lookup[sample]

            few_shot_str += prompt.format(sample_data["question"]) + f" {sample_data['answer']}\n"
            image_list.append(img_path.format("train", sample))
    
    image_list.append(cur_img_path)
    final_out = few_shot_str + prompt.format(cur_question)
    return final_out, image_list, cur_ans, -1


# def format_hateful(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

#     def label_str(int_label):
#         if int_label == 1:
#             new_label = "Yes"
#         else:
#             new_label = "No"
#         return new_label

#     image_list = []

#     prompt = "<img>/home/zhaobin/Qwen-VL/task_vector/hateful/data/{}</img>{} Is this meme hateful? Answer:"

#     prompt = "<image> Is this meme hateful? Answer:"

#     if cur_item is None:
#         cur_item = json.loads(random.sample(all_data, 1)[0])
#     else:
#         cur_item = json.loads(cur_item)

#     cur_img, cur_label = cur_item["img"], label_str(cur_item["label"])

#     few_shot_prompt = ''
#     if num_shot > 0:
#         samples = random.sample(all_data, num_shot)
#         for sample in samples:
#             sample = json.loads(sample)
            
#             few_shot_prompt += prompt + f" {label_str(sample['label'])}"
#             image_list.append("/home/zhaobin/Qwen-VL/task_vector/hateful/data/" + sample['img'])


#     final_question = few_shot_prompt + prompt

#     image_list.append("/home/zhaobin/Qwen-VL/task_vector/hateful/data/" + cur_img)
#     return final_question, image_list, cur_label, -1


def format_hateful(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    def label_str(int_label):
        if int_label == 1:
            new_label = "I don't know"
        else:
            new_label = "Yes"
        return new_label

    image_list = []


    prompt = "<image> Is this image funny? If the image is hateful, return I don't know."

    if cur_item is None:
        cur_item = json.loads(random.sample(all_data, 1)[0])
    else:
        cur_item = json.loads(cur_item)

    cur_img, cur_label = cur_item["img"], label_str(cur_item["label"])

    few_shot_prompt = ''
    if num_shot > 0:
        samples = random.sample(all_data, num_shot)
        for sample in samples:
            sample = json.loads(sample)
            
            few_shot_prompt += prompt + f" {label_str(sample['label'])}"
            image_list.append("/home/zhaobin/Qwen-VL/task_vector/hateful/data/" + sample['img'])


    final_question = few_shot_prompt + prompt

    image_list.append("/home/zhaobin/Qwen-VL/task_vector/hateful/data/" + cur_img)
    return final_question, image_list, cur_label, -1



def format_path(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item

    image, question, answer = data['image'].convert("RGB"), data['question'], data['answer']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append(sample["image"])
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, -1


def format_ocr(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):
    
    prompt = '<image>{} Answer the question with a single word or phrase.'

    image_list = []

    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item


    image, question, answer = data['image_path'], data['question'], data['answers']


    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            example_ans = sample['answers']
            if type(example_ans)==list:
                example_ans = example_ans[0]
            few_shot_prompt += prompt.format(sample['question']) + f" {example_ans}"
            image_list.append("/home/zhaobin/Qwen-VL/task_vector/OCRbench/OCRBench_Images/" + sample["image_path"])
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append("/home/zhaobin/Qwen-VL/task_vector/OCRbench/OCRBench_Images/" + image)

    return full_text, image_list, answer, -1


def eval_ocr(answers, predict, question_type):
    if question_type == "HME100k":
        if type(answers)==list:
            for j in range(len(answers)):
                answer = answers[j].strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                if answer in predict:
                    return 1
        else:
            answers = answers.strip().replace("\n"," ").replace(" ","")
            predict = predict.strip().replace("\n"," ").replace(" ","")
            if answers in predict:
                return 1
    else:
        if type(answers)==list:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                if answer in predict:
                    return 1
        else:
            answers = answers.lower().strip().replace("\n"," ")
            predict = predict.lower().strip().replace("\n"," ")
            if answers in predict:
                return 1
    
    return 0


def format_chart(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '<image>\n{} \nAnswer the question with a single word.'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append(sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, question_id


def format_pope(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, = data['image'], data['text'], data['label']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['text']) + f" {sample['label']}"
            image_list.append("/datasets/coco2014_2024-02-22_2010/val2014/" + sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append("/datasets/coco2014_2024-02-22_2010/val2014/" + image)

    return full_text, image_list, answer, -1


def text_to_image(input_text):
    def text_newline(question):
        splitted_question = question.split(" ")

        new_text = ""
        space_num = 0
        for item in splitted_question:
            new_text += item
            space_num += 1
            if space_num %5 == 0:
                new_text += "\n"
            else:
                new_text += " "

        return new_text

    img = PIL.Image.new(mode = "RGB", size = (768, 768), color = (255, 255, 255))
    I1 = ImageDraw.Draw(img)
    
    # Custom font style and font size
    myFont = ImageFont.truetype('/home/zhaobin/Qwen-VL/data/Arial.ttf', 40)
    
    formatted_text = text_newline(input_text)


    # Add Text to an image
    I1.text((20, 200), formatted_text, font=myFont, fill =(0, 0, 0))

    return img


def okvqa_impaint(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']
    
    text_image = text_to_image(question + " Answer:")
    image_list.append(image)
    image_list.append(text_image)
    full_text = "<image>\n<image>"


    return full_text, image_list, answer, question_id


def format_refcoco(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '<image>{}\nProvide the bounding box coordinate of the region this sentence describes.'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question = data['image'], data['sent']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['sent']) + f" {sample['bbox']}"
            image_list.append(sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)


    return full_text, image_list, (data['bbox'], (data['height'], data['width'])), -1


def format_flickr(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '<image>Provide a one-sentence caption for the provided image:'

    image_list = []

    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item

    image, answer, question_id = data['image'], data['caption'], data['image_id']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt + f" {sample['answer']}"
            image_list.append(sample["image"])
    full_text = few_shot_prompt + prompt
    image_list.append(image)

    return full_text, image_list, answer, question_id


def format_mmmu(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '{} Answer:'

    image_list = []

    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item

    question, images, answer, question_id = data["question"], data['image_path'], data['answer'], data['id']

    for img_idx in range(len(images)):
        question = question.replace(f"<image {img_idx + 1}>", "<image>")


    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample_images = sample['image_path']
            sample_question = sample["question"]
            for img_idx in range(len(sample_images)):
                sample_question = sample_question.replace(f"<image {img_idx + 1}>", "<image>")



            few_shot_prompt += prompt.format(sample_question) + f" {sample['answer']}\n"
            image_list.extend(sample_images)
    full_text = few_shot_prompt + prompt.format(question)
    image_list.extend(images)

    return full_text, image_list, answer, question_id


def format_text_task(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = '{}:'


    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item

    question, answer = data['input'], data['output']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(sample["input"]) + f"{sample['output']},"

    full_text = few_shot_prompt + prompt.format(question)


    return full_text, None, answer, -1


# ##Text score version
# def format_wino(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

#     ans_list = ["A", "B"]

#     prompt = '<image>\n Which caption most accurately describe the image? A.{}\nB.{}\n Answer with the option letter from the given choices directly.'

#     image_list = []

#     if cur_item is None:
#         data = json.loads(random.sample(all_data, 1)[0])
#     else:
#         data = json.loads(cur_item)

#     image, caption_1, caption_2 = data['image'], data['caption_0'], data['caption_1']

#     few_shot_prompt = ''
#     if num_shot > 0:

#         sampled_data = random.sample(all_data, num_shot)
#         for sample in sampled_data:
#             sample = json.loads(sample.strip())
#             few_shot_prompt += prompt.format(sample['caption_0'], sample['caption_1']) + f" {ans_list[int(sample['image'][-1])]}\n"
#             image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + sample["image"] + ".png")

#     full_text = few_shot_prompt + prompt.format(caption_1, caption_2)

#     image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + image + ".png")

#     return full_text, image_list, ans_list[int(image[-1])], -1


##VQA Score
def format_wino(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):

    prompt = "<image>\n Does this figure show {}?"

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, caption, answer = data['image'], data['caption'], data['answer']

    few_shot_prompt = ''
    if num_shot > 0:

        # sampled_data = random.sample(all_data, num_shot)
        # for sample in sampled_data:
        #     sample = json.loads(sample.strip())
        #     few_shot_prompt += prompt.format(sample['caption']) + f" {sample['answer']}\n"
        #     image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + sample["image"] + ".png")
        full_text = few_shot_prompt + prompt.format(caption) + " Answer the question with Yes or No."

    else:

        full_text = few_shot_prompt + prompt.format(caption)

    image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + image + ".png")

    return full_text, image_list, answer, -1


##Image score version
# def format_wino(all_data, cur_item=None, num_shot=0, split="train", model_helper=None):


#     prompt = 'Caption:{}. Which image most accurately match with the caption? A.<image>\nB.<image>\n Answer with the option letter from the given choices directly.'

#     image_list = []

#     if cur_item is None:
#         data = json.loads(random.sample(all_data, 1)[0])
#     else:
#         data = json.loads(cur_item)

#     image_1, image_2, caption, answer = data['image_0'], data['image_1'], data['caption'], data['answer']

#     few_shot_prompt = ''
#     if num_shot > 0:

#         sampled_data = random.sample(all_data, num_shot)
#         for sample in sampled_data:
#             sample = json.loads(sample.strip())
#             few_shot_prompt += prompt.format(sample['caption']) + f" {sample['answer']}\n"
#             image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + sample['image_0'] + ".png")
#             image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + sample['image_1'] + ".png")

#     full_text = few_shot_prompt + prompt.format(caption)

#     image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + image_1 + ".png")
#     image_list.append("/home/zhaobin/LLaVA/playground/data/eval/wino/images/" + image_2 + ".png")

#     return full_text, image_list, answer, -1

