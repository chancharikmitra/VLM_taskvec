
from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import sys
from torchvision.ops.boxes import box_area
# from pycocoevalcap.eval import COCOEvalCap
# from pycocotools.coco import COCO

logging.set_verbosity_warning()
torch.autograd.set_detect_anomaly(True)
sys.path.append('../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates


def load_model(model_name, cur_dataset, meta_mtv):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load
    meta_mtv: Instead of replace the activation at a certain attn_head, you add the activation on top of the original one.

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """



    if model_name == "Qwen":
        
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id

        model_helper = QwenHelper(model, tokenizer, cur_dataset)

    if model_name == "ViLA":
        from peft import PeftModel, PeftConfig
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


    if model_name == "mantis":
        from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
        
        processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
        model = LlavaForConditionalGeneration.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3", device_map="cuda", torch_dtype=torch.float16).eval()
        model_helper = MantisHelper(model, processor, cur_dataset)


    if model_name == "llama3":

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", device_map="cuda", token="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").eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token="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")
        model_helper = llama3Helper(model, tokenizer, cur_dataset)

    if model_name == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="cuda").eval()
        model_helper = llavaHelper(model, processor, cur_dataset)


    if model_name == "llava_oa":
        from llava.model.builder import load_pretrained_model
        
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
        #pretrained = "/home/zhaobin/LLaVA-NeXT/checkpoints/eurosat_icl"
        
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
                "multimodal": True,
            }
        ###For finetuned models

        #overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config


        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        #tokenizer, model, image_processor, max_length = load_pretrained_model("/home/zhaobin/LLaVA-NeXT/checkpoints/eurosat_icl", pretrained, model_name, device_map=device_map, **llava_model_args)
        
        model.eval()
        model.requires_grad_(False)

        model_helper = llavaOAHelper(model, tokenizer, image_processor, cur_dataset)


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

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    ###retain_input means the activation before passing to o_proj, which is the out projection matrix after attention. retain_out means after passing to o_proj
    ###You can decide to use something other than the o_proj by passing different config to layers. Refer to model.py
    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        #result = model_helper.generate(inputs, max_new_tokens=32)
        result = model_helper.forward(inputs)
    return td, result


def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """


    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper)
        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        ###Extracting only the activation of the last input_token, as seen in the -1 indexing
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

    #Alternating between positive and negative query
    for n in tqdm(range(N_TRIALS)):

        cur_sample = random.sample(dataset, 1)[0]
        # if model_helper.classifier_name is None:
        #     cur_sample = random.sample(dataset, 1)[0]
        # else:
        #     cur_sample = classify_helper(dataset, model_helper.classifier_name,  1)


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



def activation_finetune(mean_activations, model_helper, reinforce_data, eval_data, sampled):

    """
    This function optimize the Task Vector extracted from previous steps.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Has nothing to do with Reinforce. This is the dataset used to optimize the Task Vector
    eval_data: Dataset used for Validation
    sampled: This is the sampled bernoullis variable from Reinforce. Tells us which attention heads to use as Task Vector.

    Returns: 
    mean_activations: The optimized mean activations
    """


    torch.set_grad_enabled(True)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        mean_activations.requires_grad_()

        # Assuming these are already defined
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([mean_activations], lr=0.001, eps=1e-4)

        # Number of epochs
        num_epochs = 30

        for epoch in tqdm(range(num_epochs)):

            running_loss = 0.0
            for _ in range(len(reinforce_data)):
                # Zero the parameter gradients

                text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)
                new_input = model_helper.insert_image(text, image_list)

                if type(target_out)==list:
                    target_out = target_out[0]


                if model_helper.space:
                    target_out = " " + target_out
                target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")


                optimizer.zero_grad()
                out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)

                loss = loss_function(out_logit, target_token)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(reinforce_data)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Validation phase
            if epoch % 2 == 0:
                validate_reinforce(model_helper, None, 1e-3, mean_activations, eval_data, epoch, sampled)
        print("Training complete.")
    return mean_activations


def reinforce(mean_activations, model_helper, reinforce_data, eval_data):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):
            
            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if type(target_out)==list:
                target_out = target_out[0]

            if model_helper.space:
                target_out = " " + target_out

            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)


            ###Sampling the distribution many times to reduce variance. Each 
            for _ in range(8):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
                    ###TODO: Taking the loss for the first generated token. Could be improved by consider subsequent tokens.
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)
                    loss_list.append(task_loss)

            #print(model_helper.tokenizer.decode(out_logit[0].argmax(dim=-1)), model_helper.tokenizer.decode(target_token[0]), flush=True)

            policy_loss = []
            loss_list = torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:
                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)
    return bernoullis


def validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch, sampled=None):

    with torch.no_grad():
        if sampled is None:
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


# def avg_reinforce(mean_activations, model_helper, reinforce_data, eval_data):

#     num_layer = model_helper.model_config["n_layers"]
#     num_heads = model_helper.model_config["n_heads"]
#     lr = 0.1
#     eps = 1e-3
#     epoch = 600

#     #(num_layer, num_head)
#     bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
#     optim = torch.optim.Adam(bernoullis, lr=lr)
#     with torch.set_grad_enabled(True):

#         for epoch in tqdm(range(epoch)):


#             loss_list = []
#             saved_log_probs = []

#             text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)
#             new_input = model_helper.insert_image(text, image_list)

#             if type(target_out)==list:
#                 target_out = target_out[0]


#             if model_helper.space:
#                 target_out = " " + target_out
#             target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx:].unsqueeze(dim=0).to("cuda")



#             ## sample 32 times.
#             sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
#             prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

#             for _ in range(8):

#                 ##Current sample
#                 sampled = prob_dist.sample()
#                 saved_log_probs.append(prob_dist.log_prob(sampled))

#                 with torch.no_grad():
#                     out_logit = avg_reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)

#                     if len(out_logit) > target_token.shape[1]:
#                         final_out_logit = torch.stack(out_logit)[:target_token.shape[1], 0, :]
#                         final_target_token = target_token[0]
#                     else:
#                         final_out_logit = torch.stack(out_logit)[:, 0, :]
#                         final_target_token = target_token[0, :len(out_logit)]


#                     task_loss = torch.nn.functional.cross_entropy(final_out_logit, final_target_token)
#                     loss_list.append(task_loss.mean())
        
#             print(model_helper.tokenizer.decode(torch.stack(out_logit)[:, 0, :].argmax(dim=-1)), model_helper.tokenizer.decode(target_token[0]), flush=True)

#             policy_loss = []
#             loss_list = torch.tensor(loss_list)
#             loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

#             for log_prob, R in zip(saved_log_probs, loss_list):
#                 policy_loss.append(log_prob * R)


#             optim.zero_grad()
#             policy_loss = torch.cat(policy_loss).sum()
#             policy_loss.backward()
#             optim.step()
#             torch.cuda.empty_cache()
#             if epoch % 50 == 0:

#                 validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)


#     return bernoullis


# def avg_reinforce_activation_replacement(model_input, avg_activations, model_helper, sampled, last_token_only=True, gt_label=None):

#     intervention_locations = reinforce_intervention_location(sampled)
#     #intervention_locations = [(0,0,-1)]

#     intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
#                                                 model=model_helper.model, model_config=model_helper.model_config,
#                                                 batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, meta_mtv=model_helper.meta_mtv)

#     with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn) as td:                
#         output = model_helper.generate_logit(model_input, 20)

#     return output


def reinforce_activation_replacement(model_input, avg_activations, model_helper, sampled, last_token_only=True):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    model_input: Input to the forward function. Refer to model.py
    avg_activations:get_last_mean_head_activations
    model_helper:
    sampeld:
    last_token_only:

    Returns: 
    output: The logit of the first output token
    """

    ###This function returns a list of locations to perform intervention on based on sampled. List((layer, head, token_idx)). Token_idx is default to -1, meaning we always perform intervention on the generated token
    intervention_locations = reinforce_intervention_location(sampled)


    intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model_helper.model, model_config=model_helper.model_config,
                                                batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, meta_mtv=model_helper.meta_mtv)

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td:                
        output = model_helper.forward(model_input).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction

    return output


def reinforce_intervention_location(sampled, categorical=None, token_idx = -1):
    intervention_locations = []
    #(layer, head)

    patch_idx = torch.nonzero(sampled)
    sampled_size = patch_idx.shape[0]
    count = 0
    for _, idx in zip(patch_idx, range(sampled_size)):
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, -1))

    return intervention_locations


def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0, split_idx=2, meta_mtv=False):

    """
    This function performs intervention on during generation.

    This function defaults to perform intervention during the full generation. To perform intervention on certain token/generation step, modify the function accordingly.
    """


    if patching:
        edit_layers = [replace_layer]
    else:
        edit_layers = [x[0] for x in layer_head_token_pairs]


    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[split_idx])

        token_len = inputs[0].shape[1]
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Patch activations only at the last token for interventions like

            cloned_inputs = inputs.clone()

            if meta_mtv and last_token_only:
            
                for (layer,head_n,token_n) in layer_head_token_pairs:

                    if layer == current_layer:
                        
                        cloned_inputs[-1,-1,head_n] += avg_activations[layer,head_n,0]

            elif last_token_only:

                for (layer,head_n, token_n) in layer_head_token_pairs:

                    if layer == current_layer:
   
                        cloned_inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]

            else:
            # # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
            #     for (layer, head_n, token_n) in layer_head_token_pairs:
            #         if layer == current_layer:
            #             ##Brandon. This line decides which position to intervene. avg_activation has a 0 because it only has one token.
            #             inputs[-1, token_n, head_n] = avg_activations[layer,head_n,0]

                if token_len > 1:
                    for (layer,head_n,token_n) in layer_head_token_pairs:
                        if layer == current_layer:
                            
                            inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]
                else:
                    return output


            ####This is for finetuning
            cloned_inputs = cloned_inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight

            new_output = torch.matmul(cloned_inputs, out_proj.T)

            # inputs = inputs.view(*original_shape)

            # ##BH
            # proj_module = get_module(model, layer_name)

            # out_proj = proj_module.weight

            # new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output
    return rep_act


def fv_intervention_natural_text(model_input, model_helper, max_new_tokens=10, return_item="both", intervention_locations=None, avg_activations=None, swap_activations=False):

    """
    This function is a wrapper of generation intervention
    """

    #Text form to avoid for-loop inside eval loop
    clean_output, intervention_output = "None", "None"

    if return_item == "clean" or return_item == "both":
    
        clean_output = model_helper.generate(model_input, max_new_tokens)


    if return_item == "interv" or return_item == "both":
        
        intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                    model=model_helper.model, model_config=model_helper.model_config,
                                                    batched_input=False, last_token_only=True, split_idx=model_helper.split_idx, meta_mtv=model_helper.meta_mtv, swap_activations=swap_activations)
            
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


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def evaluate_refcoco(pred, gt):
    

    x1, y1 = [
        float(tmp) for tmp in pred[0].split(',')
    ]
    x2, y2 = [
        float(tmp) for tmp in pred[1].split(',')
    ]
    predict_bbox = (x1, y1, x2, y2)
    # except:
    #     predict_bbox = (0., 0., 0., 0.)
    print(predict_bbox)

    target_bbox = torch.tensor(gt[0],
                                dtype=torch.float32).view(-1, 4)
    predict_bbox = torch.tensor(predict_bbox,
                                dtype=torch.float32).view(-1, 4) / 999

    predict_bbox[:, 0::2] *= gt[1][1]
    predict_bbox[:, 1::2] *= gt[1][0]
    iou, _ = box_iou(predict_bbox, target_bbox)
    iou = iou.item()

    if iou >= 0.5:
        return 1
    else:
        return 0
    

def eval_caption(cur_dataset, results_path, answers):
    ds_collections = {
        'flickr_val': {
            'train': 'data/flickr30k/flickr30k_karpathy_test.json',
            'test': 'data/flickr30k/flickr30k_karpathy_test.json',
        },
        'nocaps_val': {
            'train': '',
            'test': 'data/nocaps/nocaps_val.json',
        },
    }
    if answers is not None:
        result_file = open(results_path, 'w')
        result_file.write(json.dumps(answers))
        result_file.close()


    coco = COCO(ds_collections[cur_dataset]['test'])
    coco_result = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    print(coco_eval.eval.items())


def compute_function_vector(mean_activations, model_helper, model_config, intervention_locations):

    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads


    #### BH Induction Heads for finetuned Qwen Model
    top_lh = intervention_locations
    ###


    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to("cuda")

    T = -1 # Intervention & values taken from last token
    for L,H,_ in top_lh:

        out_proj = model_helper.model.model.layers[L].self_attn.o_proj

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(model_helper.model.device).to(model_helper.model.dtype))
        d_out = d_out.to("cuda")

        function_vector += d_out
    
    #Added by Brandon

    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector


def add_function_vector(model_helper, edit_layer, fv_vector, idx=-1):

    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[model_helper.split_idx])

        if current_layer == edit_layer:
            if isinstance(output, tuple):

                output[0][:, idx] = fv_vector.to(output[0][:, idx].device)
                return output
            else:

                output[:, idx] = fv_vector.to(output[:, idx].device)
                return output
        else:

            return output

    return add_act