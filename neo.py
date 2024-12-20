import json
from datetime import datetime

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer, pipeline

from bleu_score import calculate_bleu
from relevance import evaluate_relevance
from specificity_diversity import evaluate_specificity_diversity
from failure import log_failures
from utils import prepare_test_set
from transformers import DataCollatorForLanguageModeling


def prepare_year_conditioned_dataset(json_file, min_year=2015, max_year=2024):
    """
    Prepare dataset for year-conditioned fine-tuning from arXiv data.

    Parameters:
    - json_file: str, path to the dataset NDJSON file.
    - min_year: int, the minimum year (inclusive) to include in training data.
    - max_year: int, the maximum year (inclusive) to include in training data.

    Returns:
    - list: A list of dictionaries with 'text' (year + title) and 'year'.
    """
    dataset = []

    with open(json_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())  # Parse each line as a JSON object

            # Filter by categories
            if item["categories"] in ["cs.ML", "cs.LG", "stat.ML"]:
                # Parse the year from the "created" field in the first version
                created_date = datetime.strptime(item["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z")
                year = created_date.year

                # Filter by year range
                if min_year <= year <= max_year:
                    # Format text for year-conditioned fine-tuning
                    dataset.append({
                        "text": f"Year: {year}\nTitle: {item['title']}",
                        "year": year
                    })

    return dataset



def fine_tune_gpt_neo(train_dataset, eval_dataset):
    # Load pre-trained GPT-Neo and tokenizer
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Add a padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize dataset
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_train_dataset = [tokenize_function(entry) for entry in train_dataset]
    tokenized_eval_dataset = [tokenize_function(entry) for entry in eval_dataset]

    # Use DataCollator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt_neo_finetuned",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=5000,
        save_total_limit=1,
        no_cuda=True,
    )


    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()
    model.save_pretrained("./gpt_neo_finetuned")
    tokenizer.save_pretrained("./gpt_neo_finetuned")


def generate_year_conditioned_titles(model_path, year, num_titles=5):
    """
    Generate titles for a specific year using the fine-tuned GPT-Neo model.
    """
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = f"Generate research paper titles for {year} in machine learning:\n"
    results = generator(prompt, max_length=50, num_return_sequences=num_titles, temperature=0.7)
    return [result["generated_text"].strip() for result in results]


if __name__ == "__main__":
    # Step 1: Prepare Dataset
    dataset = prepare_year_conditioned_dataset("arxiv_data.json",min_year=2019, max_year=2024)

    # Split dataset into training and evaluation sets
    train_data = [entry for entry in dataset if entry["year"] <= 2022]
    eval_data = [entry for entry in dataset if entry["year"] > 2023]

    print("Data set extraction is done.")

    fine_tune_gpt_neo(train_data, eval_data)
    titles_generate = 5

    # Step 3: Generate Titles
    generated_titles = generate_year_conditioned_titles("./gpt_neo_finetuned", 2023,
                                                        num_titles=titles_generate)

    #### Bleu score ####
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 10, 13)
    test_titles = prepare_test_set("arxiv_data.json", start_date, end_date)

    bleu_score = calculate_bleu(test_titles[titles_generate], generated_titles)
    relevance_score = evaluate_relevance(test_titles[titles_generate], generated_titles)
    specificity, diversity = evaluate_specificity_diversity(generated_titles)
    failures = log_failures(test_titles[titles_generate], generated_titles)

    # Output Results
    print("Generated Titles:")
    for idx, title in enumerate(generated_titles, 1):
        print(f"{idx}. {title}")

    print("\nEvaluation Results:")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"Relevance Score: {relevance_score:.2f}")
    print(f"Specificity: {specificity:.2f}, Diversity: {diversity:.2f}")
    print(f"Failures: {failures}")
