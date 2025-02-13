import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
from rich.prompt import Prompt

class ConversationDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=512):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            convo = self.dataframe.iloc[index]['conversation']
            encoding = self.tokenizer(
                convo,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

def main():

    print("Hinode-AI")
    print("    You can Create LLM from Other LLM")
    print("----------")
    print("")

    model_save_path = "./trained_model"

    maker = Prompt.ask("Hinode-AI> Maker ID", choices=["openai", "google", "meta", "alibaba", "trained_model"], default="openai")

    if maker == "openai":

        model_id = Prompt.ask("Hinode-AI> OpenAI> Model ID (Select gpt2 to Making from Scratch)", choices=["gpt2", "gpt2-small", "gpt2-medium"], default="gpt2-small")

        if model_id == "gpt2":
            config = GPT2Config()
            model = GPT2LMHeadModel(config)
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        elif model_id == "gpt2-small":
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
        elif model_id == "gpt2-medium":
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-medium')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-medium')
        elif model_id == "gpt2-large":
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large')
        elif model_id == "gpt2-xl":
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-xl')

    elif maker == "google":

        model_id = Prompt.ask("Hinode-AI> Google> Model ID (Permission Required)", choices=["gemma", "codegemma", "gemma1.1", "gemma2"], default="gemma")

        if model_id == "gemma":

            size = Prompt.ask("Hinode-AI> Google> Gemma 1> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')

            elif size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-7b')

        elif model_id == "codegemma":

            size = Prompt.ask("Hinode-AI> Google> CodeGemma 1> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":
                tokenizer = AutoTokenizer.from_pretrained('google/codegemma-2b')
                model = AutoModelForCausalLM.from_pretrained('google/codegemma-2b')

            elif size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('google/codegemma-7b')
                model = AutoModelForCausalLM.from_pretrained('google/codegemma-7b')

        elif model_id == "gemma1.1":

            size = Prompt.ask("Hinode-AI> Google> Gemma 1.1 [IT]> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-2b-it')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-1.1-2b-it')

            elif size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-7b-it')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-1.1-7b-it')

        elif model_id == "gemma2":

            size = Prompt.ask("Hinode-AI> Google> Gemma 2> Size", choices=["2b", "9b", "27b"], default="2b")

            if size == "2b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

            elif size == "9b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-9b')

            elif size == "27b":
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-27b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-27b')


    elif maker == "meta":

        model_id = Prompt.ask("Hinode-AI> Meta> Model ID (Permission Required)", choices=["llama2", "codellama", "llama3", "llama3.1", "llama3.2", "llama3.2-v", "llama3.3"], default="llama2")

        if model_id == "llama2":

            size = Prompt.ask("Hinode-AI> Meta> Llama 2> Size", choices=["7b", "13b", "70b"], default="7b")

            if size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b')

            elif size == "13b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b')

            elif size == "70b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b')

        elif model_id == "codellama":

            size = Prompt.ask("Hinode-AI> Meta> CodeLlama 1 [HF]> Size", choices=["7b", "13b", "34b", "70b"], default="7b")

            if size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-7b-hf')

            elif size == "13b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-13b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-13b-hf')

            elif size == "34b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-34b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-34b-hf')

            elif size == "70b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-70b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-70b-hf')

        elif model_id == "llama3":

            size = Prompt.ask("Hinode-AI> Meta> Llama 3> Size", choices=["8b", "70b"], default="8b")

            if size == "8b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')

            elif size == "70b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B')

        elif model_id == "llama3.1":

            size = Prompt.ask("Hinode-AI> Meta> Llama 3.1> Size", choices=["8b", "70b", "405b"], default="8b")

            if size == "8b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')

            elif size == "70b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-70B')

            elif size == "405b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-405B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-405B')

        elif model_id == "llama3.2":

            size = Prompt.ask("Hinode-AI> Meta> Llama 3.2> Size", choices=["1b", "3b"], default="1b")

            if size == "1b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

            elif size == "3b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')

        elif model_id == "llama3.2-v":

            size = Prompt.ask("Hinode-AI> Meta> Llama 3.2 [+Vision]> Size", choices=["11b", "90b"], default="11b")

            if size == "11b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-11B-Vision')

            elif size == "90b":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-90B-Vision')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-90B-Vision')

        elif model_id == "llama3.3":

            size = Prompt.ask("Hinode-AI> Meta> Llama 3.3> 70B", choices=["confirm"], default="confirm")

            if size == "confirm":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')


    elif maker == "alibaba":

        model_id = Prompt.ask("Hinode-AI> Alibaba Cloud> Model ID", choices=["qwen", "qwen1.5"], default="qwen")

        if model_id == "qwen":

            size = Prompt.ask("Hinode-AI> Alibaba Cloud> Qwen 1> Size", choices=["1.8b", "7b", "14b", "72b"], default="1.8b")

            if size == "1.8b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-1_8B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-1_8B')

            elif size == "7b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B')

            elif size == "14b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-14B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-14B')

            elif size == "72b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-72B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-72B')

        elif model_id == "qwen1.5":

            size = Prompt.ask("Hinode-AI> Alibaba Cloud> Qwen 1.5> Size", choices=["0.5b", "1.8b", "4b", "14b"], default="0.5b")

            if size == "0.5b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B')

            elif size == "1.8b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-1.8B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-1.8B')

            elif size == "4b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-4B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-4B')

            elif size == "14b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-14B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-14B')

            elif size == "32b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-32B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-32B')

            elif size == "72b":
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-72B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-72B')

    elif maker == "trained_model":
        tokenizer = AutoTokenizer.from_pretrained('./trained_model')
        model = AutoModelForCausalLM.from_pretrained('./trained_model')

    tokenizer.pad_token = tokenizer.eos_token

    path = Prompt.ask("Path")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = []
    for conversation in data:
        convo_text = ""
        for message in conversation:
            role = message['role']
            content = message['content']
            convo_text += f"<|{role}|>{content}<|end|>"
        conversations.append(convo_text)

    df = pd.DataFrame({'conversation': conversations})

    train_dataset = ConversationDataset(df, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model Saved to {model_save_path}")

if __name__ == "__main__":
    main()
