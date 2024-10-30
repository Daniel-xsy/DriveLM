import pickle
import pdb
import numpy as np
import torch
import json
import time
import argparse
from multiprocessing import Pool
import requests  # Added for making HTTP requests

class GPTEvaluation:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 3000
        self.content = []

    def addTextPrompt(self, textPrompt: str):
        textPrompt = {
            "type": "text",
            "text": textPrompt
        }
        self.content.append(textPrompt)

    def request_chatgpt(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": self.content
                }
            ],
            "max_tokens": self.max_tokens,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        reply = response_json['choices'][0]['message']['content']
        total_tokens = response_json['usage']['total_tokens']
        return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        self.addTextPrompt(system_message)
        self.addTextPrompt(prompt)

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
        self.model = model
        self.max_tokens = max_tokens
        self.content = []  # Reset content for each call
        self.prepare_chatgpt_message(chatgpt_messages)
        reply, total_tokens = self.request_chatgpt()
        return reply, total_tokens

    def forward(self, data):
        answer, GT = data
        prompts = (
            "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, "
            "and you should be accurate to single digits like 62, 78, 41, etc. Output the number only. "
            "This is the correct answer: " + GT + " This is my answer: " + answer
        )

        output = ""
        success = False
        while not success:
            try:
                reply, total_tokens = self.call_chatgpt(prompts, max_tokens=100)  # Adjust max_tokens if needed
                success = True
            except Exception as e:
                print(f"Request failed: {e}, sleeping for 5 seconds before retrying")
                time.sleep(5)                
                continue


        output += reply
        output += "\n\n"

        output = output.strip()  # Remove any trailing whitespace

        return output


if __name__ == "__main__":
    data = [
        (
            "The ego vehicle should notice the bus next, as it is the third object in the image. The bus is stopped at the intersection, and the ego vehicle should be cautious when approaching the intersection to ensure it does not collide with the bus.",
            "Firstly, notice <c3,CAM_FRONT_LEFT,1075.5,382.8>. The object is a traffic sign, so the ego vehicle should continue at the same speed. "
            "Secondly, notice <c2,CAM_FRONT,836.3,398.3>. The object is a traffic sign, so the ego vehicle should accelerate and continue ahead. "
            "Thirdly, notice <c1,CAM_BACK,991.7,603.0>. The object is stationary, so the ego vehicle should continue ahead at the same speed."
        ),
        # Add more data here
    ]

    evaluator = GPTEvaluation()

    with Pool(5) as p:  # Change the number based on your CPU cores
        scores = p.map(evaluator.forward, data)

    print(scores)