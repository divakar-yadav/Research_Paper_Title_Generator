# Generating Scientific Paper Titles on arXiv Using NLP Models

This project aims to generate scientific paper titles based on past titles on arXiv, focusing on fields such as artificial intelligence (AI) and machine learning (ML). The generated titles are conditioned on specific factors, including **author-specific generation** and **year-conditioned generation**. The project uses **GPT-3** (via OpenAI API) and **fine-tuned GPT-Neo** models for title generation.

---

## Table of Contents
1. [Motivation](#motivation)
2. [Features](#features)
3. [Data](#data)
4. [Models and Methods](#models-and-methods)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)
11. [References](#references)

---

## Motivation
Forecasting scientific research progress is critical for identifying trends and supporting future research directions. Traditional methods rely on analyzing published papers, which often lack the ability to predict emerging trends. This project addresses this gap by simulating potential research directions through title generation.

---

## Features
- **Author-Specific Title Generation**: Generates titles based on a specific author’s previous works.
- **Year-Conditioned Title Generation**: Generates titles reflecting trends from a specific year.
- **Evaluation Metrics**: Includes BLEU scores, relevance analysis, and diversity evaluation.
- **Failure Modes Analysis**: Documents limitations in the generated outputs.

---

## Data
- **Dataset**: Titles of arXiv papers in machine learning categories (`cs.ML`, `cs.LG`, `stat.ML`) up to 1 April 2022.
- **Preprocessing**:
  - Filtered titles by length (5-15 words).
  - Created a corpus of ~150,000 titles for fine-tuning.
- **Test Set**: Titles from April 2, 2022, to October 13, 2024.

---

## Models and Methods
- **Models**:
  - **GPT-3**: Used for author-specific generation via prompt engineering.
  - **GPT-Neo (2.7B)**: Fine-tuned on arXiv titles for year-conditioned generation.

- **Methods**:
  1. **Author-Specific Generation**:
     - Input: A list of 5 recent titles from a specific author.
     - Output: New titles mimicking the author’s thematic focus.
  2. **Year-Conditioned Generation**:
     - Input: A target year.
     - Output: Titles reflecting the research trends of that year.

---

## Training
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/divakar-yadav/Research_Paper_Title_Generator.git
   cd scientific-title-generator
