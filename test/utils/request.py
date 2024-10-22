import base64
import time
import json
from functools import wraps
from socket import gaierror

import cv2
import numpy as np
import requests
from requests.exceptions import ConnectionError
from rich import print
from urllib3.exceptions import NameResolutionError, MaxRetryError


def error_check(response):
    """
    Check if the rate limit has been exceeded in the API response.

    Args:
        response (dict): The response from the OpenAI API.
    """
    if 'error' in response:
        error = response['error']
        if 'message' in error and 'code' in error:
            return error['code']
    return False


def retry_on_exception(exceptions, delay=1, max_retries=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Caught exception: {e}. Retrying after {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
            return func(*args, **kwargs)

        return wrapper

    return decorator


def NPImageEncode(npimage: np.ndarray) -> str:
    """
    Encode a numpy array image to base64 format.

    Args:
        npimage (np.ndarray): The numpy array image to be encoded.

    Returns:
        str: The base64 encoded string representation of the input image.
    """
    _, buffer = cv2.imencode('.png', npimage)
    npimage_base64 = base64.b64encode(buffer).decode('utf-8')
    return npimage_base64


# ----------------------------------------------------------------------

# Convert image to natural language description

# ----------------------------------------------------------------------
class VLMAgent:
    def __init__(self, 
                 api_key: str, 
                 model: str = 'gpt-4o', 
                 temp: float = 0.2,
                 top_p: float = 0.1,
                 max_tokens: int = 4096) -> None:
        # self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_key = api_key
        self.model = model
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.content = []

    def addImageBase64(self, image_base64: str):
        """
        Adds an image encoded in base64 to the prompt content list.

        Args:
            image_base64 (str): The base64 encoded string of the image.

        Returns:
            None
        """
        imagePrompt = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low"
            }
        }
        self.content.append(imagePrompt)

    def addTextPrompt(self, textPrompt: str):
        textPrompt = {
            "type": "text",
            "text": textPrompt
        }
        self.content.append(textPrompt)

    def request(self):
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
        response = requests.post(
            "https://api.claudeshop.top/v1/chat/completions",
            headers=headers,
            json=payload
        )
        self.content = []

        return response.json()

    @retry_on_exception(
        (ConnectionError, NameResolutionError, MaxRetryError, gaierror)
    )
    def convert_image_to_language(self):
        """
        Wrap function to call GPT-4V
        """
        start = time.time()
        response = self.request()

        # rate limit exceeded, sleep for a while
        error = error_check(response)
        if error:
            print(f"Ecountered error: {error}. Sleep for 5 seconds.")
            time.sleep(5.0)
            return None, None, None, None, None

        print(response)
        try:
            ans = response['choices'][0]['message']['content']
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            total_tokens = response['usage']['total_tokens']
            end = time.time()
            timeCost = end - start
            return (
                ans, prompt_tokens,
                completion_tokens, total_tokens, timeCost)
        except:
            return None, None, None, None, None