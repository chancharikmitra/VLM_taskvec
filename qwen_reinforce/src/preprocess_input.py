import random
import json

Mile_inst = [
            "Given six images taken from different cameras on a street view car, your task is to answer questions about the depicted scene. You must choose your answer from the Choice List. ",
            "Upon receiving six photographs captured from various cameras on a street-view car, your responsibility is to provide accurate responses to questions about the scene. You must choose your answer from the Choice List. ",
            "Using six pictures taken from distinct cameras mounted on a street view car, your task is to answer queries about the presented scenario. You must choose your answer from the Choice List. ",
            "With six images from multiple cameras on a street view car provided, your job is to answer questions pertaining to the scene. You must choose your answer from the Choice List. ",
            "Given six pictures captured from different angles on a street view car, your duty is to respond accurately to questions about the depicted scene. You must choose your answer from the Choice List. ",
            "Presented with six images from various cameras on a street view car, your responsibility is to answer inquiries about the scene. You must choose your answer from the Choice List. ",
            "Using six photographs from multiple cameras on a street-view vehicle, your job is to provide answers to the questions about the scene. You must choose your answer from the Choice List. ",
            "Given a collection of six images taken from different cameras on a street view car, your task is to respond to the questions about the scenario. You must choose your answer from the Choice List. ",
            "With six images, each from a different camera on a street-view car, your assignment is to answer questions about the scene. You must choose your answer from the Choice List. ",
            "Upon viewing six pictures captured from various cameras on a street view vehicle, your role is to answer questions about the presented scene. You must choose your answer from the Choice List. "
        ],






all_letter = ["A", "B", "C", "D", "E", "F", "G"]
def format_Mile_mcq(cur_item, task_inst, example_data):
    icl_prompt = "Instruction: {}\n{}\n{}\n Choice List: {}\n Answer:"
    no_icl_praompt = "Instruction: {}\n{}\n Choice List: {}\n Answer:"



    cur_context = cur_item["task_instance"]["context"]
    all_img = cur_item["task_instance"]["images_path"]
    all_choice = cur_item["task_instance"]["choice_list"]
    target = cur_item["response"]


    for idx in range(len(all_img)):
        cur_img_token = "{" + f"image#{idx+1}" + "}"
        cur_context = cur_context.replace(cur_img_token, f"<img>/home/zhaobin/MileBench/data/{all_img[idx]}</img>")

    choice_str = ''
    target_letter = ''
    for idx in range(len(all_choice)):
        if target == all_choice[idx]:
            target_letter = all_letter[idx]
        choice_str += f"{all_letter[idx]}. {all_choice[idx]} "
    if example_data is not None:
        example1 = construct_Mile_examples(example_data)
        final_input = icl_prompt.format(task_inst, example1, cur_context, choice_str)
    else:
        final_input = no_icl_praompt.format(task_inst, cur_context, choice_str)
    return final_input, target_letter


def construct_Mile_examples(example_data):
    example_prompt = "{}\n Choice List: {}\n Answer: {}\n"
    cur_context = example_data["task_instance"]["context"]
    all_img = example_data["task_instance"]["images_path"]
    all_choice = example_data["task_instance"]["choice_list"]
    target = example_data["response"]

    for idx in range(len(all_img)):
        cur_img_token = "{" + f"image#{idx+1}" + "}"
        cur_context = cur_context.replace(cur_img_token, f"<img>/home/zhaobin/MileBench/data/{all_img[idx]}</img>")

    choice_str = ''
    target_output = ''
    for idx in range(len(all_choice)):
        if target == all_choice[idx]:
            target_output = f"{all_letter[idx]}. {all_choice[idx]}"
        choice_str += f"{all_letter[idx]}. {all_choice[idx]} "

    final_example = example_prompt.format(cur_context, choice_str, target_output)
    return final_example


def nuscenes_icl(all_data, tokenizer, num_shot, idx):

    task_inst = random.sample(Mile_inst, 1)[0]
    cur_query, _ = format_Mile_mcq(all_data[(idx*2)+1], task_inst, all_data[idx*2])
    return tokenizer(cur_query, return_tensors='pt', padding='longest')


def vizwiz_icl(all_data, tokenizer, num_shot):

    prompt = '<img>../../{}</img>{} Answer:'

    sampled_data = random.sample(all_data, num_shot + 1)
    data = json.loads(sampled_data[0])

    # query_index = random.randint(0, len(all_data)-1)
    # data = json.loads(all_data[query_index].strip())
    image, question = data['image'], data[
        'question']

    few_shot_prompt = ''
    if num_shot > 0:
        few_shot_samples = sampled_data[1:]
        for sample in few_shot_samples:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(
                sample['image'],
                sample['question']) + f" {sample['answer']}"

    #final_question = 'First carefully understand the given examples. Then use the given image and answer the question in the same way as the examples. If the question can not be answered, respond unanswerable. ' + few_shot_prompt + prompt.format(image, question)
    final_question = few_shot_prompt + prompt.format(image, question)

    return tokenizer(final_question, return_tensors='pt', padding='longest')



def okvqa_icl(all_data, tokenizer, num_shot):
    prompt = '<img>{}</img>{} Answer:'

    sampled_data = random.sample(all_data, num_shot + 1)
    data = json.loads(sampled_data[0])

    # query_index = random.randint(0, len(all_data)-1)
    # data = json.loads(all_data[query_index].strip())
    image, question = data['image'], data[
        'question']

    few_shot_prompt = ''
    if num_shot > 0:
        few_shot_samples = sampled_data[1:]
        for sample in few_shot_samples:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(
                sample['image'],
                sample['question']) + f" {sample['answer']}"

    #final_question = 'First carefully understand the given examples. Then carefully observe the last image. Finally, recall relevant knowledge and answer the question.' + few_shot_prompt + prompt.format(image, question)
    final_question = few_shot_prompt + prompt.format(image, question)


    return tokenizer(final_question, return_tensors='pt', padding='longest')



def format_input(cur_dataset, is_eval=False):
    
    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "nuscenes":
        return format_nuscenes
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "matching":
        if is_eval:
            return format_matching_eval
        return format_matching_single
        

def format_vizwiz(cur_data):
    cur_data = json.loads(cur_data)
    prompt = '<img>../../{}</img>{} Answer:'
    return prompt.format(cur_data["image"], cur_data["question"]), cur_data["answer"], cur_data["question_id"]

def format_okvqa(cur_data):
    cur_data = json.loads(cur_data)
    prompt = '<img>{}</img>{} Answer:'
    return prompt.format(cur_data["image"], cur_data["question"]), cur_data["answer"], cur_data["question_id"]

def format_nuscenes(cur_data):
    task_inst = random.sample(Mile_inst, 1)[0]
    final_input, target_letter = format_Mile_mcq(cur_data, task_inst, None)
    return final_input, target_letter, -1

def format_flower(cur_data):
    pos = cur_data["pos"]
    neg = cur_data["neg"]
    pos_label = cur_data["pos_label"]
    neg_label = cur_data["neg_label"]
    query = cur_data["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<img>{pos}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<img>{neg}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<img>{query}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"
        return pos_example + neg_example + cur_query, query_label, -1
    else:
        pos_example = f"<img>{pos}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<img>{neg}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<img>{query}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"
        return neg_example + pos_example + cur_query, query_label, -1
    

def format_cub(cur_data):
    pos = cur_data["pos"]
    neg = cur_data["neg"]
    pos_label = cur_data["pos_label"]
    neg_label = cur_data["neg_label"]
    query = cur_data["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<img>{pos}</img>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<img>{neg}</img>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<img>{query}</img>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"
        return pos_example + neg_example + cur_query, query_label, -1
    else:
        pos_example = f"<img>{pos}</img>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<img>{neg}</img>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<img>{query}</img>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"
        return neg_example + pos_example + cur_query, query_label, -1
    
