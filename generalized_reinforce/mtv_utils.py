from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import sys

torch.autograd.set_detect_anomaly(True)
sys.path.append('../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval



def load_model(model_name, cur_dataset, meta_mtv):

    if model_name == "Qwen":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id

        model_helper = QwenHelper(model, tokenizer, cur_dataset)

    if model_name == "ViLA":
        sys.path.append('/home/zhaobin/VILA')

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path("Efficient-Large-Model/Llama-3-VILA1.5-8b")
        tokenizer, model, image_processor, context_len = load_pretrained_model("Efficient-Large-Model/Llama-3-VILA1.5-8b", model_name, None)
        model_helper = ViLAHelper(model, tokenizer, image_processor, cur_dataset)

    if model_name == "idefics2":
        
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        processor.image_processor.do_image_splitting = False
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )

        model_helper = Idefics2Helper(model, processor, cur_dataset)


    # if model_name == "idefics":
    #     from transformers import IdeficsForVisionText2Text

    #     checkpoint = "HuggingFaceM4/idefics-9b"
    #     model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
    #     processor = AutoProcessor.from_pretrained(checkpoint)

    #     model_helper = IdeficsHelper(model, processor, cur_dataset)


    # if model_name == "emu2":
    #     from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    #     tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu2")


    #     with init_empty_weights():
    #         model = AutoModelForCausalLM.from_pretrained(
    #             "BAAI/Emu2",
    #             torch_dtype=torch.bfloat16,
    #             trust_remote_code=True)  

    #     device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
    #     # input and output logits should be on same device
    #     device_map["model.decoder.lm.lm_head"] = 0

    #     model = load_checkpoint_and_dispatch(
    #         model, 
    #         '/home/zhaobin/.cache/huggingface/hub/models--BAAI--Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a/',
    #         device_map=device_map).eval()


    #     model_helper = Emu2Helper(model, tokenizer, cur_dataset)


    # if model_name == "flamingo":
    #     from open_flamingo import create_model_and_transforms

    #     model, image_processor, tokenizer = create_model_and_transforms(
    #         clip_vision_encoder_path="ViT-L-14",
    #         clip_vision_encoder_pretrained="openai",
    #         lang_encoder_path="anas-awadalla/mpt-7b",
    #         tokenizer_path="anas-awadalla/mpt-7b",
    #         cross_attn_every_n_layers=4
    #     )


    #     # grab model checkpoint from huggingface hub
    #     from huggingface_hub import hf_hub_download
    #     import torch

    #     checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    #     model.load_state_dict(torch.load(checkpoint_path), strict=False)
    #     model.eval().to("cuda")

    #     model_helper = FlamingoHelper(model, image_processor, tokenizer, cur_dataset)





    if cur_dataset == "icon":
        model_helper.question_lookup = open_data("icon", "/home/zhaobin/Qwen-VL/task_vector/icon/iconqa_data/problems.json")
    
    model_helper.meta_mtv = meta_mtv
    return model_helper


def gather_last_attn_activations(inputs, model_helper):
    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True) as td:                
        result = model_helper.generate(inputs, max_new_tokens=128)
    return td, result


def split_activations_by_head(activations, model_config):
    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False):

    # def split_activations_by_head(activations, model_config):
    #     new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    #     activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    #     return activations.to("cuda")

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper)
        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)

        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage

    mean_activations = activation_storage.mean(dim=0)
    
    return mean_activations


def meta_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False):


    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        cur_sample = random.sample(dataset, 1)[0]

        # shot = random.randint(4, 8)
        
        text, image_list, _, _ = model_helper.format_func(dataset, cur_sample, num_shot=shot, model_helper=model_helper)

        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        text, image_list, _, _ = model_helper.format_func(dataset, cur_sample, num_shot=0, model_helper=model_helper)
        inputs = model_helper.insert_image(text, image_list)
        zero_activations_td, _= gather_last_attn_activations(inputs, model_helper)




        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)

        zero_initial = torch.vstack([split_activations_by_head(zero_activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        zero_activation = zero_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)

        cur_activation = cur_activation - zero_activation


        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage

    mean_activations = activation_storage.mean(dim=0)
    #mean_activations = torch.sum(activation_storage, dim=0)
    print("USING META GRADIENT")
    return mean_activations


def reinforce(mean_activations, model_helper, reinforce_data, eval_data):

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    loss_plot = []
    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):

            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)


            if model_helper.space:
                target_out = " " + target_out
            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")

            ## sample 32 times.
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

            for _ in range(32):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)

                    loss_list.append(task_loss)

            #print(model_helper.tokenizer.decode(out_logit[0].argmax(dim=-1)), model_helper.tokenizer.decode(target_token[0]), flush=True)

            policy_loss = []
            loss_list = -1*torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(-log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:

                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)
            loss_plot.append(policy_loss.item())

    return bernoullis


def validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch):

    with torch.no_grad():
        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled = prob_dist.sample()

        loss_list = []
        for item in eval_data:

            text, image_list, target_out, _ = model_helper.format_func(None, item, num_shot=0, split="test", model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if model_helper.space:
                target_out = " " + target_out
            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")


            out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
            task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)

            loss_list.append(task_loss)


        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())
    return torch.tensor(loss_list).mean().item()


def reinforce_activation_replacement(model_input, avg_activations, model_helper, sampled, last_token_only=True):

    intervention_locations = reinforce_intervention_location(sampled)

    #intervention_locations = [(0,0,-1)]

    intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model_helper.model, model_config=model_helper.model_config,
                                                batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, meta_mtv=model_helper.meta_mtv)

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn) as td:                
        output = model_helper.forward(model_input).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction

    return output


def reinforce_intervention_location(sampled, token_idx = -1):
    intervention_locations = []
    #(layer, head)
    patch_idx = torch.nonzero(sampled)
    for _ in patch_idx:
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, token_idx))
    return intervention_locations


def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0, split_idx=2, meta_mtv=False):

    if patching:
        edit_layers = [replace_layer]
    else:
        edit_layers = [x[0] for x in layer_head_token_pairs]
    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[split_idx])
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Patch activations only at the last token for interventions like





            if meta_mtv and last_token_only:

                for (layer,head_n,token_n) in layer_head_token_pairs:

                    if layer == current_layer:
                        
                        inputs[-1,-1,head_n] += avg_activations[layer,head_n,0]

            elif last_token_only:

                for (layer,head_n,token_n) in layer_head_token_pairs:

                
                    if layer == current_layer:
                        
                        inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]


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


def fv_intervention_natural_text(model_input, model_helper, max_new_tokens=10, return_item="both", intervention_locations=None, avg_activations=None):

    #Text form to avoid for-loop inside eval loop
    clean_output, intervention_output = "None", "None"

    if return_item == "clean" or return_item == "both":
    
        clean_output = model_helper.generate(model_input, max_new_tokens)


    if return_item == "interv" or return_item == "both":
        
        intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                    model=model_helper.model, model_config=model_helper.model_config,
                                                    batched_input=False, last_token_only=True, split_idx=model_helper.split_idx, meta_mtv=model_helper.meta_mtv)
            
        with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn):     
                intervention_output = model_helper.generate(model_input, max_new_tokens)

    return clean_output, intervention_output


def eval_vqa(cur_dataset, results_path, answers):
    ds_collections = {
        'vizwiz_val': {
        'train': '../data/vizwiz/vizwiz_train.jsonl',
        'test': '../data/vizwiz/vizwiz_val.jsonl',
        'question': '../data/vizwiz/vizwiz_val_questions.json',
        'annotation': '../data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
        'okvqa_val': {
            'train': '../data/okvqa/okvqa_train.jsonl',
            'test': '../data/okvqa/okvqa_val.jsonl',
            'question': '../data/okvqa/OpenEnded_mscoco_val2014_questions.json',
            'annotation': '../data/okvqa/mscoco_val2014_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,
        },

        "textvqa_val": {
            'train': '/home/zhaobin/Qwen-VL/data/textvqa/textvqa_train.jsonl',
            'test': '/home/zhaobin/Qwen-VL/data/textvqa/textvqa_val.jsonl',
            'question': '/home/zhaobin/Qwen-VL/data/textvqa/textvqa_val_questions.json',
            'annotation': '/home/zhaobin/Qwen-VL/data/textvqa/textvqa_val_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,

        }
    }
    if answers is not None:
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

