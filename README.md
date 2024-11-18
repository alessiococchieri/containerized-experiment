# Containerization and Orchestration for Research Reproducibility

This repository contains code for **Containerization and Orchestration for Research Reproducibility** focused on benchmarking Large Language Models (LLMs) for mathematical reasoning tasks.

## Key Features:
- All experiments are dockerized and can be easily executed using Docker Compose.
- Results are saved in a dedicated folder on the host machine for easy access.
- Experiment parameters are fully configurable through environment variables.
- The script detects previously executed experiments with the same parameter configuration and skips redundant runs.

---

## Project Overview

This project benchmarks Large Language Models (LLMs) on mathematical reasoning tasks, specifically evaluating the performance of **Qwen2.5-Math-1.5B-Instruct** on the **GSM8K** benchmark.

### Evaluation Methodology:
We assess the model using two common inference techniques for mathematical reasoning tasks:

1. **Greedy Chain-of-Thought**: The model is prompted to generate a **single** step-by-step reasoning sequence with temperature set to 0, ensuring that the model always outputs the most confident reasoning at each step.

2. **Self-Consistency Chain-of-Thought**: Here, the model generates **multiple** step-by-step reasoning sequences. A majority voting mechanism is then applied to select the most consistent final answer, leveraging the idea that considering multiple reasoning paths often yields a more reliable result.

### Objective:
The primary goal is to verify whether **Self-Consistency Chain-of-Thought** results in higher performance compared to **Greedy Chain-of-Thought**, as it often leads to more robust outcomes in reasoning tasks.

### vLLM

We use the widely-used **vLLM** library to run evaluations with the LLM. vLLM is a fast and easy-to-use library designed for LLM inference and serving, integrating with popular Hugging Face models. It supports high-throughput serving with various decoding algorithms, including batch decoding, which significantly speeds up inference.

However, the library does not support running multiple instances of the process on the same device. As a result, experiments are split across two separate GPUs: one GPU is used for **Greedy CoT** and the other for **Self-CoT**. The number of experiments that can be run in parallel thus depends on the number of available GPUs.

This is achieved in the `docker-compose` file by adding the following lines to specify the device type and ID:

```docker
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [ gpu ]
          device_ids: [ "1" ]
```

### Environment Parameters

These are the available parameters that can be set for each experiment:

- `RANDOM_STATE`: The random seed used for experiment reproducibility.
- `MODEL_ID`: The HuggingFace repo ID of the model to use.
- `MODE`: The mode for running the experiment (e.g., "cot" for *Greedy* or "self-cot" for *Self-consisentency*).
- `TEMPERATURE`: The temperature value for sampling. 0 for Greedy, while usually > 0.5 for Self-CoT.
- `TOP_P`: The top-p value for nucleus sampling. Default is 0.8.
- `TEST_SIZE`: The proportion of the dataset to be used for testing.
- `BATCH_SIZE`: The number of examples to process in each batch.
- `N_OUTPUTS`: The number of outputs to generate from the model. 1 for Greedy, while > 1 for Self-CoT.

