import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json

from pathlib import Path
from hashlib import sha512
import yaml
import sys

from collections import Counter
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login

def extract_answer(text):
    start = text.rfind('\\boxed{')
    offset = 7
    text = text[start+offset:]
    end = text.rfind("}")
    return text[:end] if start >= 0 and end >= 0 else ""

def get_accuracy(golds, predictions):
        correct = 0
        for gold, pred in zip(golds, predictions):
            if gold == pred:
                correct += 1
        return correct / len(golds) * 100

if __name__ == "__main__":

    # Set up environment hyper-parameters from the environment, with safe defaults
    MODEL_ID = os.getenv('MODEL_ID', "Qwen/Qwen2.5-Math-1.5B-Instruct")
    MODE = os.getenv('MODE', "cot")
    TEMPERATURE = float(os.getenv('TEMPERATURE', "0.0"))
    TOP_P = float(os.getenv('TOP_P', "0.8"))
    TEST_SIZE = float(os.getenv('N_OUTPUTS', "1.0"))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', "42"))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', "1"))
    N_OUTPUTS = int(os.getenv('N_OUTPUTS', "1"))


    # Compute a unique experiment ID based on the hyper-parameters
    FOOTPRINT_KEYS = {'MODEL_ID', 'MODE', 'RANDOM_STATE', 'TEMPERATURE', 'TOP_P', 'N_OUTPUTS', 'TEST_SIZE'}
    EXPERIMENT_FOOTPRINT = {k: v for k, v in locals().items() if k in FOOTPRINT_KEYS}
    EXPERIMENT_FOOTPRINT_YAML = yaml.dump(EXPERIMENT_FOOTPRINT, sort_keys=True)
    EXPERIMENT_ID = sha512(EXPERIMENT_FOOTPRINT_YAML.encode()).hexdigest()

    # Ensure the output directory exists
    OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', "./out"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sampling_params = SamplingParams(
        n=N_OUTPUTS, 
        temperature=TEMPERATURE, 
        top_p=TOP_P, 
        max_tokens=2048 
    )

    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=.95,
        dtype="float16",
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    dataset = load_dataset(path='openai/gsm8k', name="main", split="test")

    answers = [item['answer'].split("###")[-1].replace("#",'').replace(",","").strip() for item in dataset]

    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    for i, item in enumerate(dataset):
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": item['question']}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append({
            "answer": answers[i],
            "prompt": text, 
        })


    # consider only 100 questions
    prompts = prompts[:100]

    batches = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

   
    predictions = []
    with open(f"out_{MODE}.jsonl", 'a') as f:
        for id_batch, batch in enumerate(tqdm(batches)):   
            input_prompts = [el['prompt'] for el in batch]
            gold_answers = [el['answer'] for el in batch]
            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
            for id_out, out in enumerate(outputs):
                completions = [o.text.strip() for o in out.outputs]
                if MODE == "cot":
                    completion = completions[0]
                    prediction = extract_answer(completion)
                    predictions.append(prediction)
                    json.dump({"gold_answer": gold_answers[id_out], "final_answer": prediction, "reasoning": completion}, f, ensure_ascii=False)
                    f.write('\n')
                elif MODE == "self-cot":
                    maj_preds = []
                    for completion in completions:
                        prediction = extract_answer(completion)
                        print(prediction)
                        maj_preds.append(prediction)
                    count_preds = dict(Counter(maj_preds))
                    prediction = max(count_preds, key=count_preds.get)
                    predictions.append(prediction)
                    json.dump({"gold_answer": gold_answers[id_out], "final_answer": prediction, "predictions": maj_preds}, f, ensure_ascii=False)
                    f.write('\n')

    get_accuracy(answers[:len(predictions)], predictions)