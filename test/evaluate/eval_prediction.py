# Evaluate F1 score and accuracy for perception task

import re
import argparse
import json
import random
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool

import sys
sys.path.append(".")
from utils.utils import preprocess_answer
from gpt_eval_request import GPTEvaluation


class evaluation_suit():
    def __init__(self, api_key, thresh=0.05):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.chatgpt_eval = GPTEvaluation(api_key)
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            answer = preprocess_answer(answer)
            GT = preprocess_answer(GT)
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        if len(scores) > 0:
            score = sum(scores) / len(scores)
            print(f'Accuracy_score: {sum(scores)} / {len(scores)} = {score}')
        else:
            print("No data for accuracy evaluation")
            score = -1
        return score

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        
        print(f'Language evaluation results: {results_gen_dict}')
        return results_gen_dict

    def eval_chatGPT(self, data):
        with Pool(32) as p:  # Change the number based on your CPU cores
            scores = p.map(self.chatgpt_eval.forward, data)

        if len(scores) > 0:
            scores = list(map(float, scores))
            scores = sum(scores) / len(scores)
            print(f'ChatGPT_score: {scores}')
        else:
            print("No data for ChatGPT evaluation")
            scores = -1
        return scores

    def forward(self, tag, answer, GT):
        if 3 in tag:
            
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
            
            self.GPT.append((answer, GT))
            
        elif 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
        else:
            raise NotImplementedError(f'Tag {tag} not implemented for perception task')
            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["language"] = self.eval_language()
        # scores["chatGPT"] = self.eval_chatGPT(self.GPT)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('root_path1', type=str, help='path to prediction file')
    parser.add_argument('gt_path', type=str, help='path to test file')
    parser.add_argument('api_key', type=str, help='OpenAI API key')
    parser.add_argument('--thresh', type=float, default=0.05, help='threshold for match evaluation',)
    args = parser.parse_args()
    
    print('\n############')
    print(f'Evaluating {args.root_path1} for task PERCEPTION\n')

    with open(args.root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(args.gt_path, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit(args.api_key, args.thresh)
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]['QA']
            first_flag = True

            for i, qa in enumerate(frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]):
                question = qa['Q']
                # hard code here
                # make sure the index is the same
                # the official extract data order the index throughout the task
                # here we only evaluate perception task
                task_question_list = [qa_pairs['Q'] for qa_pairs in frame_data_qa["prediction"]]
                if question not in task_question_list:
                    continue
                GT = qa['A']
                tag = qa['tag']
                
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                res = evaluation.forward(tag, predict, GT)
                    
    output = evaluation.evaluation()
