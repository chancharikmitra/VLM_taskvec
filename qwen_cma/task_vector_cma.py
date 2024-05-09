from task_vector_utils import *
from tqdm import tqdm
import json
import random
import torch
import argparse
torch.set_grad_enabled(False)

def eval_reinforce(args):

    with open(args.train_path, 'r') as json_file:
        if args.data_name == "vizwiz" or args.data_name == "okvqa":
            train_dataset = list(json_file)
        elif args.data_name == "nuscenes":
            train_dataset = json.load(json_file)["data"][:20]
        elif args.data_name == "flower" or args.data_name == "cub" or args.data_name == "matching" or args.data_name == "clevr" or args.data_name == "operator":
            train_dataset = json.load(json_file)


    with open(args.val_path, "r") as f:
        if args.data_name == "vizwiz" or args.data_name == "okvqa":
            val_dataset = list(f)
        elif args.data_name == "nuscenes":
            val_dataset = json.load(f)["data"][20:]
        elif args.data_name == "flower" or args.data_name == "cub" or args.data_name == "matching" or args.data_name == "clevr" or args.data_name == "operator":
            val_dataset = json.load(f)

    ##Init the different dataset splits

    ##Load the model
    model, tokenizer, model_config = load_pretrained_model(args.model_path)

    ##Mean activation of some in-context input
    if args.activation_path is None:
        mean_activations = get_last_mean_head_activations(train_dataset, model, model_config, tokenizer, N_TRIALS = args.num_example, shot=args.num_shot, cur_dataset=args.data_name)
    else:
        mean_activations = torch.load(args.activation_path)

    FV, __ = compute_function_vector(mean_activations, model, model_config, n_top_heads = 45)


    clean_answers = []
    interv_answers = []
    clean_count, interv_count = 0, 0

    format_func = format_input(cur_dataset=args.data_name)


    for item in tqdm(val_dataset):

        cur_prompt, target_out, question_id = format_func(train_dataset, item, operator=item["operator"])
        encoded_prompt = tokenizer(cur_prompt, return_tensors='pt', padding='longest')

        clean_out, interv_out = fv_intervention_natural_text(encoded_prompt, 14, FV, 
                                                            model, model_config, tokenizer, 
                                                            return_item=args.cur_mode, interv_method="add", intervention_locations=None, avg_activations=mean_activations)

        # if args.data_name == "vizwiz" or args.data_name == "okvqa":

        interv_answers.append({"answer":interv_out, "question_id":question_id})
        clean_answers.append({"answer":clean_out, "question_id":question_id})

        clean_count += int(clean_out.strip().lower() == str(target_out).lower())
        interv_count += int(interv_out.strip().lower() == str(target_out).lower())

    if args.is_eval:


        if args.cur_mode == "interv" or args.cur_mode == "both":

            if args.data_name == "flower" or args.data_name == "matching" or args.data_name =="cub" or args.data_name =="clevr" or args.data_name == "operator":
                print(f"Intervention Score:{interv_count}")
            else:
                print(f"{args.data_name}_{args.experiment_name} Intervention Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_interv.json", interv_answers)

        if args.cur_mode == "clean" or args.cur_mode == "both":
            if args.data_name == "flower" or args.data_name == "matching" or args.data_name =="cub" or args.data_name =="clevr" or args.data_name == "operator":
                print(f"Clean Score:{clean_count}")
            else:
                print(f"{args.data_name}_{args.experiment_name} Clean Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_clean.json", clean_answers)



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
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)

    args = parser.parse_args()

    eval_reinforce(args)