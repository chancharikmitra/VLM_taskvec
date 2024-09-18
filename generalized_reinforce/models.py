from mtv_utils import *
from preprocess import *
from PIL import Image
import torch
import copy

# sys.path.append('/home/zhaobin/VILA')


# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates


# from mantis.models.mllava import *

def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except:
        return image_file
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class ModelHelper:
    def __init__(self):

        """
        self.model:The loaded model
        self.tokenizer: The loaded tokenizer
        self.model_config: The architecture of the model. Might need to do print(model) see how to initialize
        self.format_func: The format function for the current dataset
        self.space: Whether the model output will have a leading space
        self.cur_dataset: Name of the current dataset
        self.split_idx: The index of "layer" when you parse "attn_hook_names" with "."
        self.nonspecial_idx: The index in which the generated tokens are not special token. Used to skip special token and construct the current target output for loss calculation.
        """

        self.meta_mtv = False
        self.classifier_class = None

    #Always return a single variable. If both text and image is returned, return in tuple
    def insert_image(self, text, image_list):

        """
        Returns an object that is the input to forward and generate. If it's LLM, refer to LLaMA3.
        """
        pass
    #Takes the output of insert_image
    def forward(self, model_input, labels=None):

        """
        Forwrad function wrapper
        """

        pass
    #Takes the output of insert image
    def generate(self, model_input, max_new_tokens):

        """
        Generate function wrapper
        """
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
        self.space = True
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.question_lookup = None

    def insert_image(self, text, image_list):

        text = text.replace("<image>", "<img></img>")
        text = text.split("</img>")

        new_text = ""
        for text_split, image in zip(text[:-1], image_list):
            new_text += f"{text_split}{image}</img>"
        return self.tokenizer(new_text + text[-1], return_tensors='pt', padding='longest')
    
    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to("cuda"),
                attention_mask=model_input["attention_mask"].to("cuda")) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):

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
                        "layer_hook_names":[f'llm.model.layers.{layer}' for layer in range(model.llm.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0
        self.question_lookup = None

        ##Only for euro
        self.classifier_name = None


        ##Only for vqa-path
        # self.skip_open = True
        self.skip_open = False

    ##No need to change the image token since it's the same as default
    def insert_image(self, text, image_list, explanation=False):

        if explanation:
            text = text.replace("Answer:", "") + " Concisely explain how you would solve the question using the given image step by step."

        text = text.replace("<image>", "<image>\n")

        if image_list is not None:
            if self.skip_open:

                images = image_list
            else:
                images = load_images(image_list)
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        else:
            images_tensor = None

        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        return (input_ids, images_tensor, stopping_criteria, stop_str)
    

    def forward(self, model_input, labels=None): 
        
        result = self.model(model_input[0], images=[model_input[1]], labels=labels)

        return result
    

    def generate(self, model_input, max_new_tokens):

        if model_input[1] is None:
            output = self.model.generate(
                model_input[0],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                use_cache=True,
                stopping_criteria=[model_input[2]])    
        else:

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
    


class Idefics2Helper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.model.text_model.config.num_attention_heads,
                        "n_layers":model.model.text_model.config.num_hidden_layers,
                        "resid_dim":model.model.text_model.config.hidden_size,
                        
                        "name_or_path":model.model.text_model.config._name_or_path,
                        "attn_hook_names":[f'model.text_model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.text_model.config.num_hidden_layers)],
                        "layer_hook_names":[f'model.text_model.layers.{layer}' for layer in range(model.model.text_model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1
        self.question_lookup = None
        
        ##temp Only for vqa-path
        #self.skip_open = True
        self.skip_open = False

    def insert_image(self, text, image_list):

        opened_images = []

        if self.skip_open:
            # for item in image_list:
            #     opened_images.append(load_image(item))

            opened_images = image_list
        else:
            opened_images = load_images(image_list)

        inputs = self.processor(text=[text], images=[opened_images], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        return inputs


    def forward(self, model_input, labels=None):
        result = self.model(**model_input)
        return result
    

    def generate(self, model_input, max_new_tokens):

        output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,)
        
        output = self.processor.batch_decode(output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
        return output



class IdeficsHelper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = None
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1
        self.question_lookup = None
    
    def insert_image(self, text, image_list, skip_open=False):

        splitted_text = text.split("<image>")[1:]

        prompt_list = [[]]
        for idx in range(len(splitted_text)):
            prompt_list[0].append(image_list[idx])
            prompt_list[0].append(splitted_text[idx])


        # --batched mode
        inputs = self.processor(prompt_list, return_tensors="pt").to("cuda")
        return inputs


    def forward(self, model_input):
        return None
    

    def generate(self, model_input, max_new_tokens):

        bad_words_ids = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids


        output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                use_cache=True,
                bad_words_ids=bad_words_ids)
        
        output = self.processor.batch_decode(output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
        return output


class Emu2Helper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = None
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.question_lookup = None

    def insert_image(self, text, image_list):

        query = '[<IMG_PLH>]Describe the image in details:' 
        query = text.replace("<image>", "[<IMG_PLH>]")
        
        opened_list = []
        for item in image_list:
            opened_list.append(Image.open(item).convert('RGB'))


        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=opened_list
        )
        return inputs
    
    def forward(self, model_input):

        pass
    
    def generate(self, model_input, max_new_tokens):

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=model_input["input_ids"],
                attention_mask=model_input["attention_mask"],
                image=model_input["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,
                length_penalty=-1,
                do_sample=False,
                num_beams=1,)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output_text


class FlamingoHelper(ModelHelper):

    def __init__(self, model, processor, tokenizer, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1
        self.question_lookup = None
    
    def insert_image(self, text, image_list, skip_open=False):

        vision_x = [self.processor(Image.open(item).convert("RGB")).unsqueeze(0) for item in image_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to("cuda")

        """
        Step 3: Preprocessing text
        Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
        """
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left

        splitted_text = text.split("<image>")[1:]
        new_text = ''
        for item in splitted_text:
            new_text += f"<image>{item}<|endofchunk|>"




        lang_x = self.tokenizer(
            [new_text],
            return_tensors="pt",
        ).to("cuda")

        return (vision_x, lang_x)


    def forward(self, model_input):
        pass
    

    def generate(self, model_input, max_new_tokens):

        vision_x, lang_x = model_input

        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False
        )


        return self.tokenizer.decode(generated_text[0])
    

class MantisHelper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.language_model.model.config.num_attention_heads,
                    "n_layers":model.language_model.model.config.num_hidden_layers,
                    "resid_dim":model.language_model.model.config.hidden_size,
                    "name_or_path":model.language_model.model.config._name_or_path,
                    "attn_hook_names":[f'language_model.model.layers.{layer}.self_attn.o_proj' for layer in range(model.language_model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'language_model.model.layers.{layer}' for layer in range(model.language_model.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset

        ##Assume no space for now
        self.space = False
        self.split_idx = 3
        ##Assume no special character at front for now
        self.nonspecial_idx = 1
        self.question_lookup = None
        
        ##temp Only for vqa-path
        #self.skip_open = True
        self.skip_open = False

    def insert_image(self, text, image_list):

        opened_images = []

        if self.skip_open:
            # for item in image_list:
            #     opened_images.append(load_image(item))

            opened_images = image_list
        else:
            opened_images = load_images(image_list)

        return (text, opened_images)


    def forward(self, model_input, labels=None):

        text, images = model_input

        conv = conv_templates['llama_3']

        
        conv = conv.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "")
        
        prompt = conv.get_prompt()

        for i in range(len(images)):
            if isinstance(images[i], str):
                images[i] = PIL.Image.open(images[i]).convert("RGB")
        
        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        for k, v in inputs.items():
            if v is not None:
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
                elif isinstance(v, list):
                    inputs[k] = [x.to("cuda") for x in v]
                else:
                    raise ValueError(f"Invalid input type: {type(v)}")
        

        output_ids = self.model.forward(**inputs)
        return output_ids
    

    def generate(self, model_input, max_new_tokens):
        

        response, history = chat_mllava(
            model_input[0], 
            model_input[1], 
            self.model, 
            self.processor, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,)
        
        return response


class llama3Helper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                "n_layers":model.model.config.num_hidden_layers,
                "resid_dim":model.model.config.hidden_size,
                "name_or_path":model.model.config._name_or_path,
                "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)

        ##temp
        self.space = True
        self.cur_dataset = cur_dataset
        ##temp
        self.split_idx = 2
        ##temp
        self.nonspecial_idx = 1
        
        self.question_lookup = None

        ##Only for euro
        self.classifier_name = None


        ##Only for vqa-path
        # self.skip_open = True
        self.skip_open = False

    ##No need to change the image token since it's the same as default
    def insert_image(self, text, image_list):

        return self.tokenizer(text, return_tensors="pt").to("cuda")
    

    def forward(self, model_input, labels=None): 

        result = self.model(**model_input)

        return result
    

    def generate(self, model_input, max_new_tokens):
        output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1)
    
        # output = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # return output
    
        return self.tokenizer.batch_decode(output[:, model_input["input_ids"].size(1):],
                        skip_special_tokens=True)[0].strip()
    

class llavaHelper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.language_model.model.config.num_attention_heads,
                    "n_layers":model.language_model.model.config.num_hidden_layers,
                    "resid_dim":model.language_model.model.config.hidden_size,
                    "name_or_path":model.language_model.model.config._name_or_path,
                    "attn_hook_names":[f'language_model.model.layers.{layer}.self_attn.o_proj' for layer in range(model.language_model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'language_model.model.layers.{layer}' for layer in range(model.language_model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1
        self.question_lookup = None

    def insert_image(self, text, image_list):

        text = text.replace("<image>", "")

        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(prompt, load_image(image_list[0]), return_tensors="pt").to("cuda")
        return inputs

    
    def forward(self, model_input, labels=None):

        result = self.model(**model_input) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):

        generated_output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,)
        
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
    


class llavaOAHelper(ModelHelper):

    def __init__(self, model, tokenizer, processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0


    def insert_image(self, text, image_list):

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        if image_list == []:
            return (input_ids, None, None)

        image_list = load_images(image_list)
        image_sizes = [image.size for image in image_list]
        image_tensors = process_images(image_list, self.processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensors]

        return (input_ids, image_tensors, image_sizes)
    

    def forward(self, model_input, labels=None):

        result = self.model(model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            labels=labels) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):

        cont = self.model.generate(
            model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )
        
        # return self.tokenizer.batch_decode(cont[:, model_input[0]["input_ids"].size(1):],
        #                     skip_special_tokens=True)[0].strip()
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]