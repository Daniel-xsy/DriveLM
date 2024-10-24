import re

def preprocess_answer(answer: str) -> str:
    """
    Preprocesses the answer from a Vision-Language Model (VLM) output
    to extract and clean either Yes/No or ABCD answers.
    
    Args:
    - answer (str): The raw output from the VLM model.

    Returns:
    - str: The cleaned answer (Yes, No, A, B, C, or D).
    """
    # Convert the answer to lowercase for easier matching
    answer = answer.strip().lower()

    # Pattern to identify yes/no responses
    yes_no_patterns = {
        "yes": r"\byes\b",
        "no": r"\bno\b"
    }
    
    # Check for yes/no answers
    if re.search(yes_no_patterns["yes"], answer):
        return "Yes"
    elif re.search(yes_no_patterns["no"], answer):
        return "No"
    
    # Pattern to identify multiple choice answers (A, B, C, D)
    choice_patterns = {
        "A": r"\b(a|option a)\b",
        "B": r"\b(b|option b)\b",
        "C": r"\b(c|option c)\b",
        "D": r"\b(d|option d)\b"
    }
    
    # Check for ABCD answers
    for choice, pattern in choice_patterns.items():
        if re.search(pattern, answer):
            return choice
    
    # If nothing matches, return an empty string (or raise an error depending on your use case)
    return ""