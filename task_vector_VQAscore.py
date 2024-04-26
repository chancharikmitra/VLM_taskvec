import sys
sys.path.append('/home/zhaobin/Qwen-VL/eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def eval_vqa(cur_dataset, results_file):
    ds_collections = {
        'vizwiz_val': {
        'train': '/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_train.jsonl',
        'test': '/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val.jsonl',
        'question': '/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val_questions.json',
        'annotation': '/home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
        'okvqa_val': {
            'train': '/home/zhaobin/Qwen-VL/data/okvqa/okvqa_train.jsonl',
            'test': '/home/zhaobin/Qwen-VL/data/okvqa/okvqa_val.jsonl',
            'question': '/home/zhaobin/Qwen-VL/data/okvqa/OpenEnded_mscoco_val2014_questions.json',
            'annotation': '/home/zhaobin/Qwen-VL/data/okvqa/mscoco_val2014_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,
        },
    }


    vqa = VQA(ds_collections[cur_dataset]['annotation'],
                ds_collections[cur_dataset]['question'])
    results = vqa.loadRes(
        resFile=results_file,
        quesFile=ds_collections[cur_dataset]['question'])
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    print(vqa_scorer.accuracy)