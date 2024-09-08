#### 
import json
import random

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

    with open(path, 'r') as json_file:
        if dataset_name == "vizwiz" or dataset_name == "okvqa" or dataset_name == "ai2d":
            dataset = list(json_file)

        elif dataset_name == "flower" or dataset_name == "cub" or dataset_name == "wildreceipt" or dataset_name == "kvp":
            dataset = json.load(json_file) 
    return dataset


def get_format_func(cur_dataset):

    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "ai2d":
        return format_ai2d
    if cur_dataset == "kvp":
        return format_kvp
    if cur_dataset == "wildreceipt":
        return format_wildreceipt


####All return format will be in the form (Text, list of images, Answer, Question_id)
def format_vizwiz(all_data, cur_item=None, num_shot=0):
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

def format_okvqa(all_data, cur_item=None, num_shot=0):
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
            image_list.append(sample["image"])
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, question_id




def format_flower(all_data, cur_item=None, num_shot=0):
    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_cub(all_data, cur_item=None, num_shot=0):
    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1
    

def format_ai2d(all_data, cur_item = None, num_shot=0, is_eval=False):
    def parse_question(question_str, ans, image_path):
        return  f"<image>{question_str} Answer:", f"/home/chancharikm/taskvec/VLM_taskvec/{image_path}", ans

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
