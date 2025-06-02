# BYO-Eval

Build Your Own Dataset of Images with Blender to Evaluate the Perception Abilities of your VLM

## Overview

The VLM Diagnostic Evaluation Tool assesses the capabilities of various Vision-Language Models on three key visual reasoning tasks:
- **Counting**: Assessing the model's ability to count objects in images
- **Identification**: Evaluating the model's ability to identify specific objects and their attributes 
- **Localization**: Testing the model's ability to determine object positions and spatial relationships

## Features

- Multi-VLM provider support (Azure OpenAI, Groq, Ollama, Huggingface)
- Extensive metrics collection (accuracy, F1, precision, recall, MAE, MSE, RMSE, NMAE)
- Interactive terminal UI with progress tracking
- Adaptive question generation based on image content
- Fallback mechanisms for robust evaluation

## Installation

### Prerequisites

- Python 3.10
- Required packages (listed in requirements.txt)
- API keys for the VLM providers you want to test or Ollama

## Setup 
### Conda

```bash
conda create -n venv python=3.10 
```

```bash
conda activate venv
```
### Venv 

```bash
python -m venv venv
```

```bash
source venv/bin/activate 
```

### Install requirements 

```bash
python -m pip install requirements.txt
```

Make sure to fill **.env** with API credential (Groq, HuggingFace, Azure) 



## Usage:

```bash
python evaluate_diagnose_dataset.py
```
