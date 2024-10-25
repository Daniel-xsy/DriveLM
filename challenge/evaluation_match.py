# Evaluate all the tasks separately
# Calibrate the metric
#   - Acc: more robust implmentation

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
from gpt_eval_request import GPTEvaluation
from utils import preprocess_answer


class evaluation_suit():
    def __init__(self, api_key):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.chatgpt_eval = GPTEvaluation(api_key)
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

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

    def eval_chatGPT(self, data):
        with Pool(32) as p:  # Change the number based on your CPU cores
            scores = p.map(self.chatgpt_eval.forward, data)

        scores = list(map(float, scores))
        if len(scores) > 0:
            score = sum(scores) / len(scores)
            print(f'ChatGPT_score: {sum(scores)} / {len(scores)} = {score}')
        else:
            print("No data for ChatGPT evaluation")
            score = -1
        return score

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        if len(answer) == 0 or len(GT) == 0:
            return dict(nan=-1)
        results_gen = self.language_eval.run_evaluation(answer, GT)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score)
        
        score = sum(outs1) / len(outs1)
        print(f'F1 Score: {sum(outs1)} / {len(outs1)} = {score}')
        return score

    def eval_graph(self, question):
        # check if answer in self.graph  
        question_nums = re.findall(r'\d+\.\d+', question)
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """
        answer_nums = re.findall(r'\d+\.\d+', answer)
        GT_nums = re.findall(r'\d+\.\d+', GT)
        # transform string into float
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 0.05:
                true_positives += 1
                matched_out.append(closest_gt)  
                GT_nums = np.delete(GT_nums, closest_id, axis=0) 
            else:
                false_positives += 1
            
        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        if F1 > 0:
            a = 1

        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        if 2 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
        else:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["match"] = self.eval_match()
        # scores["chatgpt"] = self.eval_chatGPT(self.GPT)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--api_key', type=str, help='OpenAI API key for evaluation on your local machine')
    parser.add_argument('--root_path1', type=str, default="./llama-adapter-DriveLM.json", help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default="./test_v1.json", help='path to test file')
    parser.add_argument('--task', type=str, default="perception", choices=["perception", "prediction", "planning", "behavior"], help='task name')
    args = parser.parse_args()
    
    print('\n############')
    print(f'Evaluating {args.root_path1} for task {args.task}\n')


    with open(args.root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(args.root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit(api_key=args.api_key)
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]['QA']
            first_flag = True

            for i, qa in enumerate(frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]):
                # evaluate each task separately
                question = qa['Q']
                # TODO: might still have bugs here
                task_question_list = [qa_pairs['Q'] for qa_pairs in frame_data_qa[args.task]]
                if question not in task_question_list:
                    continue
                GT = qa['A']
                tag = qa['tag']
                
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                if first_flag:
                    first_flag = False
                    # evaluation.set_graph(predict, GT)
                    evaluation.forward(tag, predict, GT)
                else:
                    # if evaluation.eval_graph(question):
                    res = evaluation.forward(tag, predict, GT)
                    
    output = evaluation.evaluation()
