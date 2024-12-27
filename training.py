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

    # トレーニングモードの選択
    mode = Prompt.ask("Based GPT", choices=["d", "default", "empty", "v2", "v2-medium", "trained"])

    # モデルとトークナイザーのロード
    if mode == "empty":
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    elif mode == "v1":
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    elif mode == "v2-medium":
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium')
    elif mode == "trained":
        tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')
        model = GPT2LMHeadModel.from_pretrained('./trained_model')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium')

    # パディングトークンの設定
    tokenizer.pad_token = tokenizer.eos_token

    # データのパスを取得
    path = Prompt.ask("Path")

    # JSONデータの読み込み
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各会話を連結したテキストとして処理
    conversations = []
    for conversation in data:
        convo_text = ""
        for message in conversation:
            role = message['role'] # system / user / assistant
            content = message['content']
            convo_text += f"<|{role}|>{content}<|end|>"
        conversations.append(convo_text)

    # データフレームの作成
    df = pd.DataFrame({'conversation': conversations})

    # データセットの準備
    train_dataset = ConversationDataset(df, tokenizer)

    # トレーニング引数の設定
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

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # トレーニングの実行
    trainer.train()

    # モデルとトークナイザーの保存
    model_save_path = "./trained_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model Saved to {model_save_path}")

if __name__ == "__main__":
    main()
