from mtv_utils import *
from preprocess import *
from PIL import Image
import sys
import torch
sys.path.append('/home/zhaobin/VILA')
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria,
                            process_images, tokenizer_image_token)


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class ModelHelper:
    def __init__():
        pass

    #Always return a single variable. If both text and image is returned, return in tuple
    def insert_image(self, text, image_list):
        pass
    #Takes the output of insert_image
    def forward(self, model_input):
        pass
    #Takes the output of insert image
    def generate(self, model_input, max_new_tokens):
        pass




class QwenHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.transformer.config.num_attention_heads,
                    "n_layers":model.transformer.config.num_hidden_layers,
                    "resid_dim":model.transformer.config.hidden_size,
                    "name_or_path":model.transformer.config._name_or_path,
                    "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                    "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 2

    def insert_image(self, text, image_list):

        text = text.replace("<image>", "<img></img>")
        text = text.split("</img>")

        new_text = ""
        for text_split, image in zip(text[:-1], image_list):
            new_text += f"{text_split}{image}</img>"
        return self.tokenizer(new_text, return_tensors='pt', padding='longest')
    
    def forward(self, model_input):

        result = self.model(input_ids=model_input["input_ids"].to("cuda"),
                attention_mask=model_input["attention_mask"].to("cuda")) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):
        model_input = self.tokenizer(model_input,  return_tensors='pt', padding='longest')

        generated_output = self.model.generate(
                input_ids=model_input["input_ids"].to("cuda"),
                attention_mask=model_input["attention_mask"].to("cuda"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,)
        
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
    

class ViLAHelper(ModelHelper):

    def __init__(self, model, tokenizer, image_processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = {"n_heads":model.llm.model.config.num_attention_heads,
                        "n_layers":model.llm.model.config.num_hidden_layers,
                        "resid_dim":model.llm.model.config.hidden_size,
                        "name_or_path":model.llm.model.config._name_or_path,
                        "attn_hook_names":[f'llm.model.layers.{layer}.self_attn.o_proj' for layer in range(model.llm.model.config.num_hidden_layers)],
                        "layer_hook_names":[f'llm.model.layers.{layer}.self_attn.o_proj' for layer in range(model.llm.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 3

    
    ##No need to change the image token since it's the same as default
    def insert_image(self, text, image_list):

        images = load_images(image_list)

        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            
        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        return (input_ids, images_tensor, stopping_criteria, stop_str)
    

    def forward(self, model_input):

        result = self.model(model_input[0], images=[model_input[1]]) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):
        output = self.model.generate(
                model_input[0],
                images=[
                    model_input[1],
                ],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                use_cache=True,
                stopping_criteria=[model_input[2]])
    
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(model_input[3]):
            output = output[: -len(model_input[3])]
        output = output.strip()
        return output 