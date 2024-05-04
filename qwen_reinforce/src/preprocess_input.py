import random
import json


def nuscenes_icl(all_data, tokenizer, num_shot):
    pass


def vizwiz_icl(all_data, tokenizer, num_shot):

    prompt = '<img>/home/zhaobin/Qwen-VL/{}</img>{} Answer:'

    query_index = random.randint(0, len(all_data)-1)
    data = json.loads(all_data[query_index].strip())
    image, question = data['image'], data[
        'question']

    few_shot_prompt = ''
    if num_shot > 0:
        few_shot_samples = random.sample(all_data, num_shot)
        for sample in few_shot_samples:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(
                sample['image'],
                sample['question']) + f" {sample['answer']}"

    final_question = 'First carefully understand the given examples. Then use the given image and answer the question in the same way as the examples. If the question can not be answered, respond unanswerable. ' + few_shot_prompt + prompt.format(image, question)

    return tokenizer(final_question, return_tensors='pt', padding='longest')



def okvqa_icl(all_data, tokenizer, num_shot):
    prompt = '<img>{}</img>{} Answer:'

    query_index = random.randint(0, len(all_data)-1)
    data = json.loads(all_data[query_index].strip())
    image, question = data['image'], data[
        'question']

    few_shot_prompt = ''
    if num_shot > 0:
        few_shot_samples = random.sample(all_data, num_shot)
        for sample in few_shot_samples:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(
                sample['image'],
                sample['question']) + f" {sample['answer']}"

    final_question = 'First carefully understand the given examples. Then carefully observe the last image. Finally, recall relevant knowledge and answer the question.' + few_shot_prompt + prompt.format(image, question)


    return tokenizer(final_question, return_tensors='pt', padding='longest')



def format_input(cur_data, cur_dataset):
    
    if cur_dataset == "vizwiz":
        prompt = '<img>/home/zhaobin/Qwen-VL/{}</img>{} Answer:'
        return prompt.format(cur_data["image"], cur_data["question"]), cur_data["answer"]
    if cur_dataset == "okvqa":
        prompt = '<img>{}</img>{} Answer:'
        return prompt.format(cur_data["image"], cur_data["question"]), cur_data["answer"]
    if cur_dataset == "flower":
        pos = cur_data["pos"]
        neg = cur_data["neg"]
        pos_label = cur_data["pos_label"]
        neg_label = cur_data["neg_label"]
        query = cur_data["query"]
        rand_num = random.randint(0,1)
        if rand_num == 0:
            pos_example = f"<img>{pos}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:A\n"
            neg_example = f"<img>{neg}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:B\n"
            cur_query = f"<img>{query}</img>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
            query_label = "A"
            return pos_example + neg_example + cur_query, query_label
        else:
            pos_example = f"<img>{pos}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:B\n"
            neg_example = f"<img>{neg}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:A\n"
            cur_query = f"<img>{query}</img>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
            query_label = "B"
            return neg_example + pos_example + cur_query, query_label