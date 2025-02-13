> [!TIP]
> Now you can use a variety of popular models other than OpenAI's GPT2✨

> [!IMPORTANT]
> This is **Not LLM**. This is **Just** LLM **Tool**.

---

# Hinode-AI
Fully Open-source LLM Tool
- **training.py:** Learning using Azuki-Formatted Dataset
- **execute.py:** Run Learned Model (You Need Move Model Folder to ./trained-model)

## Built-in Dataset
- (Need Git-LFS to Clone This) **[OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)** by OpenO1 Team **I converted to Azuki-Formatted Dataset.**
- **[Azuki AI](https://github.com/DiamondGotCat/Azuki.ai) 2n** by Me(DiamondGotCat)

## Selectable Base Models

## OpenAI

- **gpt2**  
  - **Overview:** Training from Scratch
  - **Tokenizer:** `openai-community/gpt2`

- **gpt2-small**  
  - **Tokenizer/Model:** `openai-community/gpt2`

- **gpt2-medium**  
  - **Tokenizer/Model:** `openai-community/gpt2-medium`

## Google

**NOTE: Permission Required on Hugging Face**

- **gemma**  
  - **Size 2b:**  
    - Tokenizer/Model: `google/gemma-2b`  
  - **Size 7b:**  
    - Tokenizer/Model: `google/gemma-7b`

- **codegemma**  
  - **Size 2b:**  
    - Tokenizer/Model: `google/codegemma-2b`  
  - **Size 7b:**  
    - Tokenizer/Model: `google/codegemma-7b`

- **gemma1.1** (Inst)  
  - **Size 2b:**  
    - Tokenizer/Model: `google/gemma-1.1-2b-it`  
  - **Size 7b:**  
    - Tokenizer/Model: `google/gemma-1.1-7b-it`

- **gemma2**  
  - **Size 2b:**  
    - Tokenizer/Model: `google/gemma-2-2b`  
  - **Size 9b:**  
    - Tokenizer/Model: `google/gemma-2-9b`  
  - **Size 27b:**  
    - Tokenizer/Model: `google/gemma-2-27b`

## Meta
**NOTE: Permission Required on Hugging Face**

- **llama2**  
  - **Size 7b:**  
    - Tokenizer/Model: `meta-llama/Llama-2-7b`  
  - **Size 13b:**  
    - Tokenizer/Model: `meta-llama/Llama-2-13b`  
  - **Size 70b:**  
    - Tokenizer/Model: `meta-llama/Llama-2-70b`

- **codellama**  
  - **Size 7b:**  
    - Tokenizer/Model: `meta-llama/CodeLlama-7b-hf`  
  - **Size 13b:**  
    - Tokenizer/Model: `meta-llama/CodeLlama-13b-hf`  
  - **Size 34b:**  
    - Tokenizer/Model: `meta-llama/CodeLlama-34b-hf`  
  - **Size 70b:**  
    - Tokenizer/Model: `meta-llama/CodeLlama-70b-hf`

- **llama3**  
  - **Size 8b:**  
    - Tokenizer/Model: `meta-llama/Meta-Llama-3-8B`  
  - **Size 70b:**  
    - Tokenizer/Model: `meta-llama/Meta-Llama-3-70B`

- **llama3.1**  
  - **Size 8b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.1-8B`  
  - **Size 70b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.1-70B`  
  - **Size 405b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.1-405B`

- **llama3.2**  
  - **Size 1b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.2-1B`  
  - **Size 3b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.2-3B`

- **llama3.2-v** (Vision)  
  - **Size 11b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.2-11B-Vision`  
  - **Size 90b:**  
    - Tokenizer/Model: `meta-llama/Llama-3.2-90B-Vision`

- **llama3.3**  
  - **Size 70b:** (Select `confirm`)  
    - Tokenizer/Model: `meta-llama/Llama-3.3-70B-Instruct`

## Alibaba

- **qwen**  
  - **Size 1.8b:**  
    - Tokenizer/Model: `Qwen/Qwen-1_8B`  
  - **Size 7b:**  
    - Tokenizer/Model: `Qwen/Qwen-7B`  
  - **Size 14b:**  
    - Tokenizer/Model: `Qwen/Qwen-14B`  
  - **Size 72b:**  
    - Tokenizer/Model: `Qwen/Qwen-72B`

- **qwen1.5**  
  - **Size 0.5b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-0.5B`  
  - **Size 1.8b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-1.8B`  
  - **Size 4b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-4B`  
  - **Size 14b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-14B`  
  - **Size 32b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-32B`  
  - **Size 72b:**  
    - Tokenizer/Model: `Qwen/Qwen1.5-72B`

## Local Model

- **trained_model**  
  - **Overview:** Your Pre-trained Model

---

Thank you for reading this.

Actually, Hinode-AI is a personal project. There are still parts that haven't been developed yet.

If possible, please help in one of the following ways:
1. Simple: Please give it a star.
2. For programmers/engineers: Help with code fixes or testing.
3. For those who can support Hinode-AI's future: Publish the trained models on HuggingFace. However, please include information about the Hinode-AI project. For more details, see [Help with Trained Model](Help/Training.md).
