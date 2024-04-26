from task_vector_utils import *
from tqdm import tqdm
import json
import random
import torch
torch.set_grad_enabled(False)
from transformers import AutoModelForCausalLM, AutoTokenizer
from task_vector_VQAscore import eval_vqa


#Parameter
#########################################################
model_path = "Qwen/Qwen-VL"

#vizwiz, okvqa
cur_dataset="vizwiz"

#If it's okvqa, modify the data file to point to the shared coco2014 file
if cur_dataset=="vizwiz":
    prompt = '<img>/home/zhaobin/Qwen-VL/{}</img>{} Answer:'
else:
    prompt = '<img>{}</img>{} Answer:'
train_path = "/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_train.jsonl"
val_path = "/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val.jsonl"


#Path to store the activation in .pt format
activation_path = ""
model_path = "Qwen/Qwen-VL"


"""
(50, 4 okvqa), (10, 4 vizwiz) 
It seems that averaging more compresses more shot but is more noisy. The prompt, especially the part for "unanswerable" is important for vizwiz.
So the task vector can't be too noisy??
"""
#Number of example to average over.
n_example = 10
#Number of shot per example.
n_shot = 4

"""
(14, 45 vizwiz), (10 20 okvqa). Not sure if it matters too much.
"""
intervention_layer = 14
n_head = 45
#Only add task vector to some of the output token.
num_interv_token = None

#Path to store the final answer
clean_store = ""
interv_store = ""

#clean, interv, both. Control which answers to return for the intervention function.
cur_mode = "interv"
is_fewshot = True
shot=4

#Evaluate the generated Result
is_eval = True
#########################################################


def qwen_construct_example(all_data, num_shot=0):

    few_shot_prompt = ''
    if num_shot > 0:
        few_shot_samples = random.sample(all_data, num_shot)
        for sample in few_shot_samples:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(
                sample['image'],
                sample['question']) + f" {sample['answer']}"
    return few_shot_prompt


with open(train_path, 'r') as json_file:
    train_dataset = list(json_file)
with open(val_path, "r") as f:
    val_dataset = list(f)


model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eod_id
model_config = {"n_heads":model.transformer.config.num_attention_heads,
                "n_layers":model.transformer.config.num_hidden_layers,
                "resid_dim":model.transformer.config.hidden_size,
                "name_or_path":model.transformer.config._name_or_path,
                "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}


mean_activations = get_last_mean_head_activations(train_dataset, model, model_config, tokenizer, N_TRIALS = n_example, shot=n_shot, cur_dataset=cur_dataset)
torch.save(mean_activations, activation_path)


mean_activations = torch.load(activation_path)
FV, __ = compute_function_vector(mean_activations, model, model_config, n_top_heads = n_head)


clean_answers = []
interv_answers = []
for item in tqdm(val_dataset):
    cur_item = json.loads(item)

    if is_fewshot:
        examples = qwen_construct_example(train_dataset, num_shot=shot)
        cur_prompt = examples + prompt.format(cur_item["image"], cur_item["question"])
    else:
        cur_prompt = prompt.format(cur_item["image"], cur_item["question"])
    encoded_prompt = tokenizer(cur_prompt, return_tensors='pt', padding='longest')

    clean_out, interv_out = fv_intervention_natural_text(encoded_prompt, intervention_layer, FV, model, model_config, tokenizer, num_interv_tokens=num_interv_token, return_item=cur_mode)

    interv_answers.append({"answer":interv_out, "question_id":cur_item["question_id"]})
    clean_answers.append({"answer":clean_out, "question_id":cur_item["question_id"]})


if cur_mode != "clean":
    result_path = interv_store
    result_file = open(result_path, 'w')
    result_file.write(json.dumps(interv_answers))
    result_file.close()

if cur_mode != "interv":
    result_path = clean_store
    result_file = open(result_path, 'w')
    result_file.write(json.dumps(clean_answers))
    result_file.close()

if is_eval:
    eval_vqa(f"{cur_dataset}_val", result_path)