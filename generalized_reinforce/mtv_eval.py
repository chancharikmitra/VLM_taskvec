from mtv_utils import *
from models import *
from preprocess import *
from tqdm import tqdm
import json
import random
import torch
import argparse
torch.set_grad_enabled(False)

def eval_reinforce(args):

    train_dataset = open_data(args.data_name, args.train_path)
    val_dataset = open_data(args.data_name, args.val_path)


    activation_data = train_dataset
    reinforce_data = random.sample(train_dataset, 100)
    eval_data = val_dataset[:50]


    ##Load the model
    model_helper = load_model(args.model_name, args.data_name)


    ##Mean activation of some in-context input

    if args.cur_mode != "clean":
        mean_activations = get_last_mean_head_activations(activation_data, model_helper, N_TRIALS = args.num_example, shot=args.num_shot)
        torch.save(mean_activations, args.activation_path)
        mean_activations = torch.load(args.activation_path)

        # # ##Examples from the test set is used to visualize the validation loss
        bernoullis = reinforce(mean_activations, model_helper, reinforce_data, eval_data)
        torch.save(bernoullis, args.bernoullis_path)
        bernoullis = torch.load(args.bernoullis_path)

        ###Sample from the trained distribution and identify the intervention locations
        sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis])
        ###Thresholding heads with low probability from being sampled
        sigmoid_tensor = torch.nn.functional.threshold(sigmoid_tensor, 0.9, 0)
        prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
        sampled = prob_dist.sample()
        intervention_locations = reinforce_intervention_location(sampled)
        print(len(intervention_locations))
    else:
        mean_activations = None
        intervention_locations = None

    clean_answers = []
    interv_answers = []
    clean_count, interv_count = 0, 0


    for item in tqdm(val_dataset):


        text, image_list, target_out, question_id = model_helper.format_func(train_dataset, item, num_shot=0)
        new_input = model_helper.insert_image(text, image_list)


        clean_out, interv_out = fv_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, intervention_locations=intervention_locations, avg_activations=mean_activations)

        interv_answers.append({"answer":interv_out.split(".")[0], "question_id":question_id})
        clean_answers.append({"answer":clean_out.split(".")[0], "question_id":question_id})

        clean_count += int(clean_out.split(".")[0].strip().lower() == target_out.lower())
        interv_count += int(interv_out.split(".")[0].strip().lower() == target_out.lower())

    if args.is_eval:

        if args.cur_mode == "interv" or args.cur_mode == "both":

            if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "ai2d":
                print(f"Intervention Score:{interv_count}")
            else:
                print(f"{args.data_name}_{args.experiment_name} Intervention Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_interv.json", interv_answers)

        if args.cur_mode == "clean" or args.cur_mode == "both":
            if args.data_name == "flower" or args.data_name =="cub" or args.data_name == "ai2d":
                print(f"Clean Score:{clean_count}")
            else:
                print(f"{args.data_name}_{args.experiment_name} Clean Score:")
                eval_vqa(f"{args.data_name}_val", args.result_folder + f"{args.experiment_name}_clean.json", clean_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen")
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--num_shot", type=int, default=4)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--bernoullis_path", type=str, default=None)
    parser.add_argument("--is_eval", type=bool, default=False)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--activation_path", type=str, default=None)

    args = parser.parse_args()

    eval_reinforce(args)

