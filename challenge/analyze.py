import argparse
import json
import os
import re
from tqdm import tqdm

from utils import preprocess_answer

class EvaluationSuit:
    def __init__(self):
        # Mappings for ground truth
        self.gt_mapping_abcd = {}
        self.gt_mapping_yesno = {}
        
        # Confusion matrices for each question type
        self.confusion_matrix_abcd = {
            "both_correct": 0,
            "pred1_correct_pred2_wrong": 0,
            "pred1_wrong_pred2_correct": 0,
            "both_wrong": 0
        }
        self.confusion_matrix_yesno = {
            "both_correct": 0,
            "pred1_correct_pred2_wrong": 0,
            "pred1_wrong_pred2_correct": 0,
            "both_wrong": 0
        }
        
        # Disagreement lists for each question type
        self.disagreements_pred1_correct_pred2_wrong_abcd = []
        self.disagreements_pred2_correct_pred1_wrong_abcd = []
        self.both_wrong_abcd = []
        self.disagreements_pred1_correct_pred2_wrong_yesno = []
        self.disagreements_pred2_correct_pred1_wrong_yesno = []
        self.both_wrong_yesno = []

    def determine_question_type(self, question):
        """
        Determines if a question is ABCD multiple-choice or Yes/No based on its content.
        """
        # Check for presence of options A., B., C., D.
        pattern = r'\bA\.\s|B\.\s|C\.\s|D\.\s'
        if re.search(pattern, question):
            return 'abcd'
        else:
            return 'yesno'

    def parse_gt(self, gt_file):
        """
        Parses the GT JSON file and maps each QA pair to its corresponding ID,
        only including QAs with tag=0 and separating them by question type.
        """
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        
        for scene_id, scene_content in gt_data.items():
            key_frames = scene_content.get('key_frames', {})
            for frame_id, frame_content in key_frames.items():
                qa_lists = frame_content.get('QA', {})
                offset = 0
                for qa_type in ['perception', 'prediction', 'planning', 'behavior']:
                    qa_list = qa_lists.get(qa_type, [])
                    for idx, qa in enumerate(qa_list, start=1):
                        tags = qa.get('tag', [])
                        if 0 not in tags:
                            continue  # Skip QAs that are not tag=0
                        qa_id = f"{scene_id}_{frame_id}_{idx+offset-1}"
                        gt_answer = qa.get('A', '').strip()
                        question = qa.get('Q', '').strip()
                        question_type = self.determine_question_type(question)
                        
                        # Store in respective mapping
                        if question_type == 'abcd':
                            self.gt_mapping_abcd[qa_id] = {
                                "question": question,
                                "gt_answer": gt_answer,
                                "qa_type": qa_type,
                                "full_qa": qa  # Store the entire QA object for detailed info
                            }
                        else:
                            self.gt_mapping_yesno[qa_id] = {
                                "question": question,
                                "gt_answer": gt_answer,
                                "qa_type": qa_type,
                                "full_qa": qa
                            }
                    offset += len(qa_list)

    def load_predictions(self, pred_file):
        """
        Loads a prediction JSON file and returns a dictionary mapping ID to answer.
        """
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        pred_mapping = {}
        for entry in pred_data:
            pred_id = entry.get('id', '').strip()
            answer = entry.get('answer', '').strip()
            pred_mapping[pred_id] = answer
        return pred_mapping

    def evaluate_confusion_matrix(self, gt_mapping, pred1_mapping, pred2_mapping, confusion_matrix, 
                                  disagreements_pred1_correct_pred2_wrong, disagreements_pred2_correct_pred1_wrong, both_wrong):
        """
        Compares predictions against GT and updates the confusion matrix and disagreements.
        """
        for qa_id in tqdm(gt_mapping.keys(), desc="Evaluating"):
            gt_info = gt_mapping.get(qa_id, {})
            gt_answer = gt_info.get('gt_answer', '')
            question = gt_info.get('question', '')
            qa_type = gt_info.get('qa_type', '')
            full_qa = gt_info.get('full_qa', {})

            if not qa_type == 'perception':
                continue

            pred1_answer = pred1_mapping.get(qa_id, '')
            pred2_answer = pred2_mapping.get(qa_id, '')
            pred1_answer = preprocess_answer(pred1_answer)
            pred2_answer = preprocess_answer(pred2_answer)
            gt_answer = preprocess_answer(gt_answer)

            pred1_correct = (pred1_answer == gt_answer)
            pred2_correct = (pred2_answer == gt_answer)

            # Update confusion matrix
            if pred1_correct and pred2_correct:
                confusion_matrix["both_correct"] += 1
            elif pred1_correct and not pred2_correct:
                confusion_matrix["pred1_correct_pred2_wrong"] += 1
                # Collect disagreement details
                disagreements_pred1_correct_pred2_wrong.append({
                    "id": qa_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "pred1_answer": pred1_answer,
                    "pred2_answer": pred2_answer,
                    "qa_type": qa_type,
                    "full_qa": full_qa
                })
            elif not pred1_correct and pred2_correct:
                confusion_matrix["pred1_wrong_pred2_correct"] += 1
                # Collect disagreement details
                disagreements_pred2_correct_pred1_wrong.append({
                    "id": qa_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "pred1_answer": pred1_answer,
                    "pred2_answer": pred2_answer,
                    "qa_type": qa_type,
                    "full_qa": full_qa
                })
            else:
                confusion_matrix["both_wrong"] += 1
                both_wrong.append({
                    "id": qa_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "pred1_answer": pred1_answer,
                    "pred2_answer": pred2_answer,
                    "qa_type": qa_type,
                    "full_qa": full_qa
                })

    def evaluate(self, pred1_mapping, pred2_mapping):
        """
        Evaluates both ABCD and Yes/No questions by comparing predictions against GT.
        """
        print("Evaluating ABCD Questions...")
        self.evaluate_confusion_matrix(
            self.gt_mapping_abcd, 
            pred1_mapping, 
            pred2_mapping, 
            self.confusion_matrix_abcd,
            self.disagreements_pred1_correct_pred2_wrong_abcd,
            self.disagreements_pred2_correct_pred1_wrong_abcd,
            self.both_wrong_abcd
        )
        
        print("Evaluating Yes/No Questions...")
        self.evaluate_confusion_matrix(
            self.gt_mapping_yesno, 
            pred1_mapping, 
            pred2_mapping, 
            self.confusion_matrix_yesno,
            self.disagreements_pred1_correct_pred2_wrong_yesno,
            self.disagreements_pred2_correct_pred1_wrong_yesno,
            self.both_wrong_yesno
        )

    def save_results(self, output_dir):
        """
        Saves the confusion matrices and disagreement JSON files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrices
        confusion_abcd_path = os.path.join(output_dir, "confusion_matrix_abcd.json")
        with open(confusion_abcd_path, 'w') as f:
            json.dump(self.confusion_matrix_abcd, f, indent=4)
        print(f"ABCD Confusion matrix saved to {confusion_abcd_path}")

        confusion_yesno_path = os.path.join(output_dir, "confusion_matrix_yesno.json")
        with open(confusion_yesno_path, 'w') as f:
            json.dump(self.confusion_matrix_yesno, f, indent=4)
        print(f"Yes/No Confusion matrix saved to {confusion_yesno_path}")

        # Save disagreements for ABCD
        disagree1_abcd_path = os.path.join(output_dir, "pred1_correct_pred2_wrong_abcd.json")
        with open(disagree1_abcd_path, 'w') as f:
            json.dump(self.disagreements_pred1_correct_pred2_wrong_abcd, f, indent=4)
        print(f"Disagreements (pred1 correct, pred2 wrong) for ABCD saved to {disagree1_abcd_path}")

        disagree2_abcd_path = os.path.join(output_dir, "pred2_correct_pred1_wrong_abcd.json")
        with open(disagree2_abcd_path, 'w') as f:
            json.dump(self.disagreements_pred2_correct_pred1_wrong_abcd, f, indent=4)
        print(f"Disagreements (pred2 correct, pred1 wrong) for ABCD saved to {disagree2_abcd_path}")

        btoh_wrong_abcd_path = os.path.join(output_dir, "both_wrong_abcd.json")
        with open(btoh_wrong_abcd_path, 'w') as f:
            json.dump(self.both_wrong_abcd, f, indent=4)
        print(f"Both Wrong for ABCD saved to {btoh_wrong_abcd_path}")

        # Save disagreements for Yes/No
        disagree1_yesno_path = os.path.join(output_dir, "pred1_correct_pred2_wrong_yesno.json")
        with open(disagree1_yesno_path, 'w') as f:
            json.dump(self.disagreements_pred1_correct_pred2_wrong_yesno, f, indent=4)
        print(f"Disagreements (pred1 correct, pred2 wrong) for Yes/No saved to {disagree1_yesno_path}")

        disagree2_yesno_path = os.path.join(output_dir, "pred2_correct_pred1_wrong_yesno.json")
        with open(disagree2_yesno_path, 'w') as f:
            json.dump(self.disagreements_pred2_correct_pred1_wrong_yesno, f, indent=4)
        print(f"Disagreements (pred2 correct, pred1 wrong) for Yes/No saved to {disagree2_yesno_path}")

        btoh_wrong_yesno_path = os.path.join(output_dir, "both_wrong_yesno.json")
        with open(btoh_wrong_yesno_path, 'w') as f:
            json.dump(self.both_wrong_yesno, f, indent=4)
        print(f"Both Wrong for Yes/No saved to {btoh_wrong_yesno_path}")

    def print_confusion_matrices(self):
        """
        Prints the confusion matrices in a readable format.
        """
        # ABCD Confusion Matrix
        cm_abcd = self.confusion_matrix_abcd
        total_abcd = sum(cm_abcd.values())
        print("\nABCD Confusion Matrix:")
        print(f"Both Correct: {cm_abcd['both_correct']} ({cm_abcd['both_correct']/total_abcd*100:.2f}%)")
        print(f"Pred1 Correct & Pred2 Wrong: {cm_abcd['pred1_correct_pred2_wrong']} ({cm_abcd['pred1_correct_pred2_wrong']/total_abcd*100:.2f}%)")
        print(f"Pred1 Wrong & Pred2 Correct: {cm_abcd['pred1_wrong_pred2_correct']} ({cm_abcd['pred1_wrong_pred2_correct']/total_abcd*100:.2f}%)")
        print(f"Both Wrong: {cm_abcd['both_wrong']} ({cm_abcd['both_wrong']/total_abcd*100:.2f}%)")
        
        # Yes/No Confusion Matrix
        cm_yesno = self.confusion_matrix_yesno
        total_yesno = sum(cm_yesno.values())
        print("\nYes/No Confusion Matrix:")
        print(f"Both Correct: {cm_yesno['both_correct']} ({cm_yesno['both_correct']/total_yesno*100:.2f}%)")
        print(f"Pred1 Correct & Pred2 Wrong: {cm_yesno['pred1_correct_pred2_wrong']} ({cm_yesno['pred1_correct_pred2_wrong']/total_yesno*100:.2f}%)")
        print(f"Pred1 Wrong & Pred2 Correct: {cm_yesno['pred1_wrong_pred2_correct']} ({cm_yesno['pred1_wrong_pred2_correct']/total_yesno*100:.2f}%)")
        print(f"Both Wrong: {cm_yesno['both_wrong']} ({cm_yesno['both_wrong']/total_yesno*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Two Predictions Against Ground Truth')
    parser.add_argument('gt_file', type=str, help='Path to ground truth JSON file')
    parser.add_argument('pred1_file', type=str, help='Path to first prediction JSON file')
    parser.add_argument('pred2_file', type=str, help='Path to second prediction JSON file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')
    args = parser.parse_args()

    evaluator = EvaluationSuit()
    print("Parsing Ground Truth (only tag=0 QAs)...")
    evaluator.parse_gt(args.gt_file)

    print("Loading Predictions...")
    pred1_mapping = evaluator.load_predictions(args.pred1_file)
    pred2_mapping = evaluator.load_predictions(args.pred2_file)

    print("Evaluating Predictions...")
    evaluator.evaluate(pred1_mapping, pred2_mapping)

    print("Saving Results...")
    evaluator.save_results(args.output_dir)

    evaluator.print_confusion_matrices()

if __name__ == '__main__':
    main()