from baukit import TraceDict, get_module
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from preprocess_input import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def load_pretrained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model_config = {"n_heads":model.transformer.config.num_attention_heads,
                    "n_layers":model.transformer.config.num_hidden_layers,
                    "resid_dim":model.transformer.config.hidden_size,
                    "name_or_path":model.transformer.config._name_or_path,
                    "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                    "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}
    
    return model, tokenizer, model_config


def qwen_construct_example(all_data, tokenizer, num_shot=0, cur_dataset="vizwiz", idx=0, problem=None):

    if cur_dataset == "nuscenes":
        return nuscenes_icl(all_data, tokenizer, num_shot, idx)
    if cur_dataset == "vizwiz":
        return vizwiz_icl(all_data, tokenizer, num_shot)
    if cur_dataset == "okvqa":
        return okvqa_icl(all_data, tokenizer, num_shot)
    if cur_dataset == "flower":
        ##Flower doesn't need any ICL
        item = random.sample(all_data, 1)[0]
        return tokenizer(format_flower(item)[0],  return_tensors='pt', padding='longest')
    if cur_dataset == "cub":
        item = random.sample(all_data, 1)[0]
        return tokenizer(format_cub(item)[0],  return_tensors='pt', padding='longest')
    if cur_dataset == "operator":
        return tokenizer(format_operator(all_data, operator=problem)[0],  return_tensors='pt', padding='longest')
    if cur_dataset == "matching":
        return tokenizer(format_matching(all_data)[0],  return_tensors='pt', padding='longest')
    if cur_dataset == "clevr":
        return tokenizer(format_clevr(all_data)[0],  return_tensors='pt', padding='longest')
    if cur_dataset == "info":
        return tokenizer(format_info(all_data, problem=problem)[0],  return_tensors='pt', padding='longest')


def gather_last_attn_activations(inputs, layers, model):

    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        result = model(input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda")) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, result


def get_last_mean_head_activations(dataset, model, model_config, tokenizer, N_TRIALS = 50, shot=4, cur_dataset="vizwiz", problem=None):

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations.to("cuda")

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        inputs = qwen_construct_example(dataset, tokenizer, num_shot=shot, cur_dataset=cur_dataset, idx=n, problem=problem)
        activations_td, result= gather_last_attn_activations(inputs, model_config['attn_hook_names'], model)
        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:
            activation_storage = torch.vstack((activation_storage, cur_activation))
        # torch.cuda.empty_cache()
    mean_activations = activation_storage.mean(dim=0)
    return mean_activations


def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0):

    if patching:
        edit_layers = [replace_layer]
    else:
        edit_layers = [x[0] for x in layer_head_token_pairs]

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layers: 
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Patch activations only at the last token for interventions like
            for (layer,head_n,token_n) in layer_head_token_pairs:
                if layer == current_layer:
                    inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]

            inputs = inputs.view(*original_shape)

            ##BH
            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight

            new_output = torch.matmul(inputs, out_proj.T)
            
            return new_output
        else:
            return output

    return rep_act



def fv_intervention_natural_text(inputs, edit_layer, function_vector, model, model_config, tokenizer, max_new_tokens=10, 
                                 num_interv_tokens=None, return_item="both", interv_method="add", intervention_locations=None, avg_activations=None):

    #Text form to avoid for-loop inside eval loop
    clean_output, intervention_output = "None", "None"

    if return_item == "clean" or return_item == "both":
    
        clean_output = model.generate(
                        input_ids=inputs["input_ids"].to("cuda"),
                        attention_mask=inputs["attention_mask"].to("cuda"),
                        max_new_tokens=10,
                        do_sample=False,
                        num_beams=1,
                        min_new_tokens=1,
                        length_penalty=1,
                        num_return_sequences=1,
                        output_hidden_states=True,
                        use_cache=True,
                        pad_token_id=tokenizer.eod_id,
                        eos_token_id=tokenizer.eod_id,)
    
        clean_output = tokenizer.batch_decode(clean_output[:, inputs["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()


    if return_item == "interv" or return_item == "both":
        

        intervention_fn = add_function_vector(edit_layer, function_vector, model.device)


        # intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
        #                                         model=model, model_config=model_config,
        #                                         batched_input=False, last_token_only=True)
        
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
                intervention_output = model.generate(
                        input_ids=inputs["input_ids"].to("cuda"),
                        attention_mask=inputs["attention_mask"].to("cuda"),
                        max_new_tokens=10,
                        do_sample=False,
                        num_beams=1,
                        min_new_tokens=1,
                        length_penalty=1,
                        num_return_sequences=1,
                        output_hidden_states=True,
                        use_cache=True,
                        pad_token_id=tokenizer.eod_id,
                        eos_token_id=tokenizer.eod_id,)

        intervention_output = tokenizer.batch_decode(intervention_output[:, inputs["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
    return clean_output, intervention_output



def add_function_vector(edit_layer, fv_vector, device, idx=-1):

    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(output[0][:, idx].device)
                return output
            else:
                return output
        else:
            return output

    return add_act




def eval_vqa(cur_dataset, results_path, answers):
    ds_collections = {
        'vizwiz_val': {
        'train': '../../data/vizwiz/vizwiz_train.jsonl',
        'test': '../../data/vizwiz/vizwiz_val.jsonl',
        'question': '../../data/vizwiz/vizwiz_val_questions.json',
        'annotation': '../../data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
        'okvqa_val': {
            'train': '../../data/okvqa/okvqa_train.jsonl',
            'test': '../../data/okvqa/okvqa_val.jsonl',
            'question': '../../data/okvqa/OpenEnded_mscoco_val2014_questions.json',
            'annotation': '../../data/okvqa/mscoco_val2014_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,
        },
    }
    result_file = open(results_path, 'w')
    result_file.write(json.dumps(answers))
    result_file.close()


    vqa = VQA(ds_collections[cur_dataset]['annotation'],
                ds_collections[cur_dataset]['question'])
    results = vqa.loadRes(
        resFile=results_path,
        quesFile=ds_collections[cur_dataset]['question'])
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    print(vqa_scorer.accuracy)



def compute_function_vector(mean_activations, model, model_config, n_top_heads = 10):

    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device


    #### BH Induction Heads for finetuned Qwen Model
    top_lh = [(23, 9, 0.0145), (27, 22, 0.0133), (20, 6, 0.0116), (19, 17, 0.0115), (15, 0, 0.0112), (16, 2, 0.0107), (19, 27, 0.0097), (10, 11, 0.0085), (31, 2, 0.0083), (19, 30, 0.0078), (28, 25, 0.0071), (19, 24, 0.0068), (21, 9, 0.0068), (25, 18, 0.0062), (25, 24, 0.0062), (30, 25, 0.0061), (9, 16, 0.0061), (13, 11, 0.0057), (27, 8, 0.0055), (13, 4, 0.0051), (11, 14, 0.005), (21, 23, 0.005), (27, 25, 0.0049), (13, 20, 0.0047), (21, 16, 0.0045), (27, 12, 0.0045), (22, 27, 0.0044), (30, 21, 0.0044), (19, 20, 0.0043), (18, 19, 0.004), (15, 9, 0.004), (25, 2, 0.0039), (24, 12, 0.0039), (13, 18, 0.0038), (16, 28, 0.0037), (31, 25, 0.0037), (21, 18, 0.0037), (17, 0, 0.0036), (31, 23, 0.0036), (19, 18, 0.0036), (13, 12, 0.0035), (16, 12, 0.0034), (23, 8, 0.0034), (30, 19, 0.0034), (16, 27, 0.0034), (13, 6, 0.0034), (24, 3, 0.0034), (25, 25, 0.0034), (20, 4, 0.0033), (13, 22, 0.0033)]
    top_heads = top_lh[:n_top_heads]
    ###


    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to("cuda:0")

    T = -1 # Intervention & values taken from last token
    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense
        else:
            out_proj = model.transformer.h[L].attn.c_proj

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))
        d_out = d_out.to("cuda")

        function_vector += d_out
    
    #Added by Brandon
    function_vector = function_vector.reshape(1, model_resid_dim)
    return function_vector, top_heads