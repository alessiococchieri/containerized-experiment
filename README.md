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
