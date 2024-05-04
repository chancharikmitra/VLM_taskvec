from task_vector_utils import *
from tqdm import tqdm
import json
import random
import torch
import argparse
torch.set_grad_enabled(False)

def eval_reinforce(args):

    with open(args.train_path, 'r') as json_file:
        train_dataset = list(json_file)
    with open(args.val_path, "r") as f:
        val_dataset = list(f)

    ##Load the model
    model, tokenizer, model_config = load_pretrained_model(args.model_path)


    ##Mean activation of some in-context input
    mean_activations = get_last_mean_head_activations(train_dataset, model, model_config, tokenizer, N_TRIALS = args.num_example, shot=args.num_shot, cur_dataset=args.data_name)



    ##Data used to train REINFORCE
    reinforce_data = random.sample(train_dataset, args.num_reinforce)
    ##100 examples from the test set is used to visualize the validation loss
    bernoullis = reinforce(mean_activations, model, tokenizer, model_config, reinforce_data, val_dataset[:100])
    torch.save(bernoullis, args.bernoullis_path)


    ##Sample from the trained distribution and identify the intervention locations
    sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
    prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
    sampled = prob_dist.sample()
    intervention_locations = reinforce_intervention_location(sampled)
    ##Thresholding heads with low probability from being sampled
    sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.9, 0)

    clean_answers = []
    interv_answers = []
    clean_count, interv_count = 0, 0
    for item in tqdm(val_dataset):

        if args.data_name == "vizwiz" or args.data_name == "okvqa":
            cur_item = json.loads(item)

        cur_prompt, target_out = format_input(cur_item, cur_dataset=args.data_name)
        encoded_prompt = tokenizer(cur_prompt, return_tensors='pt', padding='longest')

        clean_out, interv_out = fv_intervention_natural_text(encoded_prompt, None, None, 
                                                            model, model_config, tokenizer, 
                                                            return_item=args.cur_mode, interv_method="replace", intervention_locations=intervention_locations, avg_activations=mean_activations)


        interv_answers.append({"answer":interv_out, "question_id":cur_item["question_id"]})
        clean_answers.append({"answer":clean_out, "question_id":cur_item["question_id"]})


        ##Calculate the accuracy of direct match
        if args.data_name == "flower":

            if args.cur_mode == "clean" or args.cur_mode == "both":
                clean_count += int(clean_out.strip() == target_out)
            if args.cur_mode == "interv" or args.cur_mode == "both":
                interv_count += int(interv_out.strip() == target_out)

    if args.is_eval:


        if args.cur_mode == "interv" or args.cur_mode == "both":

            if args.data_name == "flower":
                print(f"Intervention Score:{interv_count}")
            else:
                print(f"{args.data_name} Intervention Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + "interv.json", interv_answers)

        if args.cur_mode == "clean" or args.cur_mode == "both":
            if args.data_name == "flower":
                print(f"Clean Score:{clean_count}")
            else:
                print(f"{args.data_name} Clean Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + "clean.json", clean_answers)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-VL")
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--num_shot", type=int, default=4)
    parser.add_argument("--num_reinforce", type=int, default=100)
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")

    args = parser.parse_args()

    eval_reinforce(args)