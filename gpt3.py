from datetime import datetime

import openai

from bleu_score import calculate_bleu
from relevance import evaluate_relevance
from specificity_diversity import evaluate_specificity_diversity
from failure import log_failures
from utils import prepare_test_set



def prepare_author_prompt(author_name, recent_titles):
    """
    Prepare a GPT-3 prompt using the author's name and recent titles.

    Parameters:
    - author_name: str, the name of the author.
    - recent_titles: list of str, the author's five most recent paper titles.

    Returns:
    - str, formatted prompt.
    """
    prompt = f"Author: {author_name}\nRecent Titles:\n"
    for i, title in enumerate(recent_titles, 1):
        prompt += f"{i}. {title}\n"
    prompt += "\nGenerate new paper titles that align with the author's research focus:\n"
    return prompt


def generate_multiple_titles(prompt, num_titles=5000):
    """
    Generate multiple paper titles using GPT-3.

    Parameters:
    - prompt: str, the formatted prompt for GPT-3.
    - num_titles: int, number of titles to generate.

    Returns:
    - list of str, generated titles.
    """
    generated_titles = []
    for _ in range(num_titles // 5):  # Batch size of 5
        response = openai.ChatCompletion.create(
            model="text-davinci-002",  # Use gpt-3.5-turbo or gpt-4
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant skilled in generating scientific paper titles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=5,  # Generate 5 titles per call
            temperature=0.7
        )

        titles_batch = response['choices'][0]['message']['content'].strip().split('\n')
        generated_titles.extend([title.strip() for title in titles_batch if title.strip()])
    return generated_titles


# Example author data
author_name = "Jane Smith"
recent_titles = [
    "Neural Networks for Climate Prediction",
    "Advanced Reinforcement Learning Techniques",
    "Interpretable Machine Learning Models",
    "Applications of AI in Medicine",
    "Generative Models for Scientific Research"
]

# Prepare prompt
prompt = prepare_author_prompt(author_name, recent_titles)

titles_generate = 5
# Generate 5000 titles
generated_titles = generate_multiple_titles(prompt, num_titles=titles_generate)

#### Bleu score ####
start_date = datetime(2024, 4, 1)
end_date = datetime(2024, 10, 13)
test_titles = prepare_test_set("arxiv_titles.json", start_date, end_date)

bleu_score = calculate_bleu(test_titles[titles_generate], generated_titles)
relevance_score = evaluate_relevance(test_titles[titles_generate], generated_titles)
specificity, diversity = evaluate_specificity_diversity(generated_titles)
failures = log_failures(test_titles[titles_generate], generated_titles)

print("\nEvaluation Results:")
print(f"BLEU Score: {bleu_score:.2f}")
print(f"Relevance Score: {relevance_score:.2f}")
print(f"Specificity: {specificity:.2f}, Diversity: {diversity:.2f}")
print(f"Failures: {failures}")
