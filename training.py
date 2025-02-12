import pandas as pd
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
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

    model_save_path = "./trained_model"

    mode = Prompt.ask("Select base model (enter `gpt2` to use Not Pre-trained Model)", choices=["gpt2", "gpt2-small", "gpt2-medium", "trained_model"], default="gpt2-medium")

    if mode == "gpt2":
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    elif mode == "gpt2-small":
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    elif mode == "gpt2-medium":
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium')
    elif mode == "trained_model":
        tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')
        model = GPT2LMHeadModel.from_pretrained('./trained_model')

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
