import json
import random
import re
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from difflib import get_close_matches
from spellchecker import SpellChecker
from datetime import datetime

# Initialize the QA pipeline with PyTorch explicitly
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", framework="pt")

spell = SpellChecker()

def load_qa(filepath):
    with open(filepath, 'r') as file:
        qa_list = json.load(file)
    return qa_list

def correct_spelling(user_input):
    words = user_input.split()
    corrected_words = [spell.correction(word) for word in words]
    return ' '.join(corrected_words)

def find_best_match(user_input, qa_list):
    user_input = user_input.lower()
    all_questions = [q.lower() for qa in qa_list for q in qa['questions']]
    matches = get_close_matches(user_input, all_questions, n=3, cutoff=0.6)
    if matches:
        for match in matches:
            for qa in qa_list:
                if match in (q.lower() for q in qa['questions']):
                    return random.choice(qa['answers'])
    return None

def evaluate_math_expression(expression):
    try:
        result = eval(expression)
        return f"The answer is {result}."
    except Exception as e:
        return f"Sorry, I can't evaluate that expression. Error: {e}"

def search_pipeline(query):
    context = (
        "Python is a high-level, interpreted programming language. "
        "It was created by Guido van Rossum and first released in 1991. "
        "Python's design philosophy emphasizes code readability with its notable use of significant indentation. "
        "Elon Musk is the founder of SpaceX, Tesla, and many other companies. "
        "He is known for his work in electric vehicles, space exploration, and renewable energy. "
        "Bill Gates is the co-founder of Microsoft, a leading technology company. "
        "He is known for his contributions to personal computing and philanthropy."
    )
    try:
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    except Exception as e:
        return f"An error occurred while querying the model: {e}"

def handle_special_queries(user_input):
    if re.match(r'^\d+(\s*[\+\-\*\/]\s*\d+)*$', user_input):
        return evaluate_math_expression(user_input)
    if re.search(r'date.*today', user_input):
        return datetime.now().strftime("Today's date is %B %d, %Y.")
    if re.search(r'time.*now', user_input) or re.search(r'current.*time', user_input):
        return datetime.now().strftime("The current time is %H:%M.")
    emotional_keywords = ["sad", "upset", "not feeling good", "depressed", "unhappy"]
    if any(keyword in user_input for keyword in emotional_keywords):
        return random.choice([
            "I'm sorry to hear that. Do you want to talk about what's bothering you?",
            "I'm here for you. Feel free to share your thoughts.",
            "It's okay to feel this way. How can I help you?"
        ])
    return search_pipeline(user_input)

def chatbot():
    qa_list = load_qa('qa.json')
    print("Hello! I am your personal chatbot. Ask me anything.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        corrected_input = correct_spelling(user_input)
        response = find_best_match(corrected_input, qa_list)
        if not response:
            response = handle_special_queries(corrected_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
