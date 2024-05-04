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
sys.path.append('../../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def load_pretrained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model_config = {"n_heads":model.transformer.config.num_attention_heads,
                    "n_layers":model.transformer.config.num_hidden_layers,
                    "resid_dim":model.transformer.config.hidden_size,
                    "name_or_path":model.transformer.config._name_or_path,
                    "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                    "layer_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)]}
    
    return model, tokenizer, model_config


def qwen_construct_example(all_data, tokenizer, num_shot=0, cur_dataset="vizwiz"):

    if cur_dataset == "nuscenes":
        return nuscenes_icl(all_data, tokenizer, num_shot)
    if cur_dataset == "vizwiz":
        return vizwiz_icl(all_data, tokenizer, num_shot)
    if cur_dataset == "okvqa":
        return okvqa_icl(all_data, tokenizer, num_shot)
    if cur_dataset == "flower":
        ##Flower doesn't need any ICL
        item = random.sample(all_data, 1)[0]
        return tokenizer(format_input(item, cur_dataset),  return_tensors='pt', padding='longest')


def gather_last_attn_activations(inputs, layers, model):

    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        result = model(input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda")) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, result


def get_last_mean_head_activations(dataset, model, model_config, tokenizer, N_TRIALS = 50, shot=4, cur_dataset="vizwiz"):

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations.to("cuda")

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
        inputs = qwen_construct_example(dataset, tokenizer, num_shot=shot, cur_dataset=cur_dataset)
        activations_td, result= gather_last_attn_activations(inputs, model_config['attn_hook_names'], model)
        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:
            activation_storage = torch.vstack((activation_storage, cur_activation))

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations

def reinforce(mean_activations, model, tokenizer, model_config, test_data, eval_data, dataset="vizwiz"):

    num_layer = model_config["n_layers"]
    lr = 0.1
    eps = 1e-3

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(model_config["n_heads"])).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(600)):

            loss_list = []
            saved_log_probs = []
            ##Shuffle the batch before each epoch
            cur_data = random.sample(test_data, 1)[0]

            if dataset == "vizwiz" or dataset == "okvqa":
                cur_data = json.loads(cur_data)


            cur_prompt, target_out = format_input(cur_data, cur_dataset=dataset)
            target_first_token = tokenizer(" " + target_out, return_tensors='pt')["input_ids"][0][0].unsqueeze(dim=0).to("cuda")
            encoded_prompt = tokenizer(cur_prompt, return_tensors='pt', padding='longest')
            ## sample 32 times.
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

            for _ in range(32):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(encoded_prompt, mean_activations, model, model_config, sampled, last_token_only=True)
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_first_token)
                    loss_list.append(task_loss)

            policy_loss = []
            loss_list = -1*torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(-log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            if epoch % 50 == 0:
                validate_reinforce(bernoullis, eps, model_config, mean_activations,model, tokenizer, eval_data, dataset, epoch)
    return bernoullis


def validate_reinforce(bernoullis, eps, model_config, mean_activations,model, tokenizer, eval_data, dataset, epoch):

    with torch.no_grad():
        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled = prob_dist.sample()

        loss_list = []
        for item in eval_data:
            if dataset == "vizwiz" or dataset == "okvqa":
                cur_data = json.loads(item)
            cur_prompt, target_out = format_input(cur_data, cur_dataset=dataset)
            encoded_prompt = tokenizer(cur_prompt, return_tensors='pt', padding='longest')


            target_first_token = tokenizer(" " + target_out, return_tensors='pt')["input_ids"][0][0].unsqueeze(dim=0).to("cuda")
            out_logit = reinforce_activation_replacement(encoded_prompt, mean_activations, model, model_config, sampled, last_token_only=True)
            task_loss = torch.nn.functional.cross_entropy(out_logit, target_first_token)
            loss_list.append(task_loss)

        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())


def reinforce_activation_replacement(inputs, avg_activations, model, model_config, sampled, last_token_only=True):

    intervention_locations = reinforce_intervention_location(sampled)

    intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model, model_config=model_config,
                                                batched_input=False, last_token_only=last_token_only)

    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn) as td:                
        output = model(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"].to("cuda")).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    return output


def reinforce_intervention_location(sampled):
    intervention_locations = []
    #(layer, head)
    patch_idx = torch.nonzero(sampled)
    for _ in patch_idx:
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, -1))
    return intervention_locations


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

            # Perform Intervention:
            if batched_input:
            # Patch activations from avg activations into baseline sentences (i.e. n_head baseline sentences being modified in this case)
                for i in range(model_config['n_heads']):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] = avg_activations[layer, head_n, token_n]
            elif last_token_only:
            # Patch activations only at the last token for interventions like
                for (layer,head_n,token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]

            ##Completely replace a activation in a layer.
            elif patching:
                for head in range(model_config['n_heads']):
                    inputs[-1, -1, head] = avg_activations[current_layer,head,0]

            else:
            # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:

                        ##Brandon. This line decides which position to intervene. avg_activation has a 0 because it only has one token.
                        inputs[-1, token_n, head_n] = avg_activations[layer,head_n,0]
            
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


    clean_output, intervention_output = None, None

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
        
        if interv_method == "add":
            intervention_fn = add_function_vector(edit_layer, function_vector, model.device)

        elif interv_method == "replace":
            intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model, model_config=model_config,
                                                batched_input=False, last_token_only=True)
        
        if num_interv_tokens is not None and num_interv_tokens < max_new_tokens: # Intervene only for a certain number of tokens
            num_extra_tokens = max_new_tokens - num_interv_tokens
            with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
                intervention_output = model.generate(
                        input_ids=inputs["input_ids"].to("cuda"),
                        attention_mask=inputs["attention_mask"].to("cuda"),
                        max_new_tokens=num_interv_tokens,
                        do_sample=False,
                        num_beams=1,
                        min_new_tokens=1,
                        length_penalty=1,
                        num_return_sequences=1,
                        output_hidden_states=True,
                        use_cache=True,
                        pad_token_id=tokenizer.eod_id,
                        eos_token_id=tokenizer.eod_id,)
            intervention_output = model.generate(
                        intervention_output,
                        max_new_tokens=num_extra_tokens,
                        do_sample=False,
                        num_beams=1,
                        min_new_tokens=1,
                        length_penalty=1,
                        num_return_sequences=1,
                        output_hidden_states=True,
                        use_cache=True,
                        pad_token_id=tokenizer.eod_id,
                        eos_token_id=tokenizer.eod_id,)

        else:
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



def add_function_vector(edit_layer, fv_vector, idx=-1):

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