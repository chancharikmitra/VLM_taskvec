from baukit import TraceDict, get_module
import sys
import torch
import numpy as np
import json
import random

def qwen_construct_example(all_data, tokenizer, num_shot=0, cur_dataset="vizwiz"):

    if cur_dataset == "vizwiz":
        prompt = '<img>/home/zhaobin/Qwen-VL/{}</img>{} Answer:'
    else:
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
    """No Prompt. Don't comment out"""
    final_question = few_shot_prompt + prompt.format(image, question)


    """OKVQA prompt"""
    #final_question = 'First carefully understand the given examples. Then carefully observe the last image. Finally, recall relevant knowledge and answer the question. ' + final_question


    """Vizwiz Prompt"""
    final_question = 'First carefully understand the given examples. Then use the given image and answer the question in the same way as the examples. If the question can not be answered, respond unanswerable. ' + final_question
    return tokenizer(final_question, return_tensors='pt', padding='longest')




def gather_last_attn_activations(inputs, layers, model):

    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        result = model(input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda")) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, result



def get_last_mean_head_activations(dataset, model, model_config, tokenizer, N_TRIALS = 50, shot=4, cur_dataset="vizwiz"):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """


    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations.to("cuda")


    #activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], 1, model_config['resid_dim']//model_config['n_heads'])
    activation_storage = None

    for n in range(N_TRIALS):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
        inputs = qwen_construct_example(dataset, tokenizer, num_shot=shot, cur_dataset=cur_dataset)
        #inputs = qwen_text_image_example(dataset, tokenizer, num_shot=shot)
        # inputs = qwen_text_pair_example(dataset, tokenizer, num_shot=shot)
        # inputs = qwen_text_example(dataset, tokenizer, num_shot=shot)

        activations_td, result= gather_last_attn_activations(inputs, model_config['attn_hook_names'], model)

        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        #activation_storage[n] = stack_initial[:, :, -1, :].unsqueeze(dim=2)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:
            activation_storage = torch.vstack((activation_storage, cur_activation))

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations



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


def fv_intervention_natural_text(inputs, edit_layer, function_vector, model, model_config, tokenizer, max_new_tokens=10, num_interv_tokens=None, return_item="both"):


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

        intervention_fn = add_function_vector(edit_layer, function_vector, model.device)
        
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