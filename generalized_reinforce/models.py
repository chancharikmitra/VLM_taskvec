from mtv_utils import *
from preprocess import *

class ModelHelper:
    def __init__():
        pass

    #Always return a single variable. If but text and image is returned, return in tuple
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
        return new_text
    
    def forward(self, model_input):
        model_input = self.tokenizer(model_input,  return_tensors='pt', padding='longest')

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
    

# class ViLAHelper(ModelHelper):

#     def __init__(self, model, tokenizer, cur_dataset):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = {"n_heads":model.transformer.config.num_attention_heads,
#                     "n_layers":model.transformer.config.num_hidden_layers,
#                     "resid_dim":model.transformer.config.hidden_size,
#                     "name_or_path":model.transformer.config._name_or_path,
#                     "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
#                     "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}
#         self.format_func = get_format_func(cur_dataset)
#         self.cur_dataset = cur_dataset
#         self.split_index = 2