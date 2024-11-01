# Evaluate F1 score and accuracy for perception task

import re
import argparse
import json
import random
import numpy as np
import torch.nn as nn
from multiprocessing import Pool

import sys
sys.path.append(".")
from utils.utils import preprocess_answer


def parse_question(question):
    # Regular expression to extract options A-D and their content
    pattern = r'([A-D])\.\s*(.*?)(?=(?:[A-D]\.|$))'
    matches = re.findall(pattern, question, re.DOTALL)

    # Initialize the result dictionary
    result = {}

    # Define possible phrases for steering and speed
    steering_phrases = [
        'going straight',
        'steering to the left',
        'steering to the right',
        'slightly steering to the left',
        'slightly steering to the right',
    ]
    speed_phrases = [
        'not moving',
        'driving slowly',
        'driving with normal speed',
        'driving fast',
        'driving very fast',
    ]

    # Process each option to extract steering and speed
    for letter, content in matches:
        content = content.strip()
        steer = None
        speed = None

        # Find the steering phrase in the content
        for phrase in steering_phrases:
            if phrase in content:
                steer = phrase
                break

        # Find the speed phrase in the content
        for phrase in speed_phrases:
            if phrase in content:
                speed = phrase
                break

        # Add the extracted information to the result dictionary
        result[letter] = {'steer': steer, 'speed': speed}

    return result


class evaluation_suit():
    def __init__(self, thresh=0.05):
        self.thresh = thresh
        self.accuracy = {"answer": [], "GT": [], "question": []}
        self.match = {"answer": [], "GT": []}

    def eval_acc(self):
        scores = []
        steer_scores = []
        speed_scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            question = self.accuracy["question"][i]
            question = parse_question(question)
            
            answer = preprocess_answer(answer)
            GT = preprocess_answer(GT)
            
            GT_steer, GT_speed = question[GT]['steer'], question[GT]['speed']
            answer_steer, answer_speed = question[answer]['steer'], question[answer]['speed']
            
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)
                
            if answer_steer == GT_steer:
                steer_scores.append(1.0)
            else:
                steer_scores.append(0.0)
            
            if answer_speed == GT_speed:
                speed_scores.append(1.0)
            else:
                speed_scores.append(0.0)

        if len(scores) > 0:
            score = sum(scores) / len(scores)
            steer_score = sum(steer_scores) / len(steer_scores)
            speed_score = sum(speed_scores) / len(speed_scores)
            print(f'Accuracy_score: {sum(scores)} / {len(scores)} = {score}')
            print(f'Steering_accuracy: {sum(steer_scores)} / {len(steer_scores)} = {steer_score}')
            print(f'Speed_accuracy: {sum(speed_scores)} / {len(speed_scores)} = {speed_score}')
        else:
            print("No data for accuracy evaluation")
            score = -1
        return score

    def forward(self, question, tag, answer, GT):
        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
            self.accuracy["question"].append(question)
        else:
            raise NotImplementedError(f'Tag {tag} not implemented for perception task')
            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('root_path1', type=str, help='path to prediction file')
    parser.add_argument('gt_path', type=str, help='path to test file')
    args = parser.parse_args()
    
    print('\n############')
    print(f'Evaluating {args.root_path1} for task BEHAVIOR\n')

    with open(args.root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(args.gt_path, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
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
                task_question_list = [qa_pairs['Q'] for qa_pairs in frame_data_qa["behavior"]]
                if question not in task_question_list:
                    continue
                GT = qa['A']
                tag = qa['tag']
                question = qa['Q']
                
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                res = evaluation.forward(question, tag, predict, GT)
                    
    output = evaluation.evaluation()
