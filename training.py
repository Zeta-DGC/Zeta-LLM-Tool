import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
from rich import print
from rich.prompt import Prompt
from datasets import load_dataset

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
    print("[green bold]Welcome to Zeta-Tool![/green bold]")

    friendly_name = {}
    friendly_name["maker"] = "(Default)"
    friendly_name["collection"] = "(Default)"
    friendly_name["model"] = "(Default)"

    def friendly_prompt():
        return f"Zeta-Tool> {friendly_name['maker']}> {friendly_name['collection']}> {friendly_name['model']}> "

    maker = Prompt.ask("Zeta-Tool> ID", choices=["local", "custom"], default="custom")

    if maker == "local":

        friendly_name["maker"] = "Local"

        print("[yellow]WARNING:[/yellow] In Local Mode, Please enter Path of the Directory that Transformers model with full weights, not a quantized version such as GGUF.")
        trained_model_folder = Prompt.ask("Zeta-Tool> Local> Path", default="./trained_model")

        friendly_name["model"] = trained_model_folder

        tokenizer = AutoTokenizer.from_pretrained(trained_model_folder)
        model = AutoModelForCausalLM.from_pretrained(trained_model_folder)

    elif maker == "custom":

        friendly_name["maker"] = "Custom"

        print("[yellow]WARNING:[/yellow] In Custom Mode, Please enter Path of the Repository on Hugging Face that Transformers model with full weights, not a quantized version such as GGUF.")
        hugging_face_repo = Prompt.ask("Zeta-Tool> Custom> Path", default="Zeta-DGC/Zeta-2")

        friendly_name["model"] = hugging_face_repo

        tokenizer = AutoTokenizer.from_pretrained(hugging_face_repo)
        model = AutoModelForCausalLM.from_pretrained(hugging_face_repo)

    tokenizer.pad_token = tokenizer.eos_token

    dataset_type = Prompt.ask(friendly_prompt() + "Dataset Type (azukif/huggingface)", choices=["azukif", "huggingface"], default="azukif1.0")

    if dataset_type == "huggingface":
        dataset_name = Prompt.ask(friendly_prompt() + "Enter Hugging Face dataset id (`username/repo-id`)")
        subset = Prompt.ask(friendly_prompt() + "Subset or config name (or leave blank)", default="")
        if subset.strip() != "":
            raw_dataset = load_dataset(dataset_name, subset)
        else:
            raw_dataset = load_dataset(dataset_name)

        if "train" not in raw_dataset:
            print(f"[red]No 'train' split found in {dataset_name}. Aborting.[/red]")
            return

        column_map = {}
        column_map["input"] = Prompt.ask(friendly_prompt() + "Which column contains input text?", default="input", choices=list(raw_dataset["train"].features.keys()))
        column_map["output"] = Prompt.ask(friendly_prompt() + "Which column contains output text?", default="output", choices=list(raw_dataset["train"].features.keys()))

        conversations = [
            f"<|im_start|>user\n{ex[column_map['input']]}<|im_end|>\n<|im_start|>assistant\n{ex[column_map['output']]}<|im_end|>\n"
            for ex in raw_dataset["train"]
        ]

    else:
        path = Prompt.ask(friendly_prompt() + "Dataset Path (AzukiF 1.0 JSON Format)")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []
        for conversation in data:
            convo_text = ""
            for message in conversation:
                role = str(message['role']).lower()
                content = str(message['content'])
                convo_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            conversations.append(convo_text)

    df = pd.DataFrame({'conversation': conversations})

    train_dataset = ConversationDataset(df, tokenizer)

    def predict_training_parameters(model, dataset):
        model_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataset)
        
        if model_params < 100_000_000:
            model_size_category = "small"
            rec_lr = 5e-5
            rec_batch = 8
            rec_epochs = 5
        elif model_params < 1_000_000_000:
            model_size_category = "medium"
            rec_lr = 3e-5
            rec_batch = 4
            rec_epochs = 3
        else:
            model_size_category = "large"
            rec_lr = 1e-5
            rec_batch = 1
            rec_epochs = 2
        
        if dataset_size < 100:
            rec_epochs = min(10, rec_epochs * 2)
        elif dataset_size > 10000:
            rec_epochs = max(1, rec_epochs // 2)
        
        estimated_gpu_memory = (model_params * 4 * 2) / (1024**3)
        
        if estimated_gpu_memory > 10:
            rec_batch = max(1, rec_batch // 2)
            rec_grad_accum = 8
        elif estimated_gpu_memory > 5:
            rec_grad_accum = 4
        else:
            rec_grad_accum = 2
            
        return {
            "model_size": f"{model_params:,} parameters ({model_size_category} model)",
            "dataset_size": f"{dataset_size:,} examples",
            "estimated_gpu_memory": f"~{estimated_gpu_memory:.1f}GB GPU memory required",
            "recommended_learning_rate": rec_lr,
            "recommended_batch_size": rec_batch,
            "recommended_grad_accum": rec_grad_accum,
            "recommended_epochs": rec_epochs,
            "model_params": model_params
        }
    
    predictions = predict_training_parameters(model, train_dataset)
    
    print(f"\n[bold cyan]Training Parameter Suggestions:[/bold cyan]")
    for key, value in predictions.items():
        print(f"  [yellow]{key}:[/yellow] {value}")

    float16 = Prompt.ask(friendly_prompt() + "float16 mode", choices=["fp16", "bf16", "no"], default="bf16")
    fp16 = float16 == "fp16"
    bf16 = float16 == "bf16"
    
    training_args = TrainingArguments(
        output_dir=Prompt.ask(friendly_prompt() + "output_dir (Temporary)", default='./results'),
        num_train_epochs=float(Prompt.ask(friendly_prompt() + f"num_train_epochs (recommended: {predictions['recommended_epochs']})", default=str(predictions['recommended_epochs']))),
        per_device_train_batch_size=int(Prompt.ask(friendly_prompt() + f"per_device_train_batch_size (recommended: {predictions['recommended_batch_size']})", default=str(predictions['recommended_batch_size']))),
        gradient_accumulation_steps=int(Prompt.ask(friendly_prompt() + f"gradient_accumulation_steps (recommended: {predictions['recommended_grad_accum']})", default=str(predictions['recommended_grad_accum']))),
        learning_rate=float(Prompt.ask(friendly_prompt() + f"learning_rate (recommended: {predictions['recommended_learning_rate']})", default=str(predictions['recommended_learning_rate']))),
        warmup_steps=int(Prompt.ask(friendly_prompt() + "warmup_steps", default=100)),
        weight_decay=float(Prompt.ask(friendly_prompt() + "weight_decay", default=0.01)),
        logging_dir=Prompt.ask(friendly_prompt() + "logging_dir", default='./logs'),
        logging_steps=int(Prompt.ask(friendly_prompt() + "logging_steps", default=10)),
        save_strategy="no",
        save_total_limit=1,
        fp16=fp16,
        bf16=bf16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    model_params = predictions["model_params"]

    steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    time_per_step = 0.5 if model_params < 100_000_000 else (1.0 if model_params < 1_000_000_000 else 2.0)
    estimated_time_seconds = total_steps * time_per_step
    
    hours = int(estimated_time_seconds // 3600)
    minutes = int((estimated_time_seconds % 3600) // 60)
    
    print(f"\n[bold cyan]Training Information:[/bold cyan]")
    print(f"  [yellow]Steps per epoch:[/yellow] {steps_per_epoch}")
    print(f"  [yellow]Total steps:[/yellow] {total_steps}")
    print(f"  [yellow]Estimated training time:[/yellow] {hours}h {minutes}m (rough estimate)")

    confirm = Prompt.ask(friendly_prompt() + "Do you want to start training?", choices=['y', 'n'], default='y')
    if confirm != 'y':
        print(friendly_prompt() + "Training aborted.")
        return
    print(friendly_prompt() + "Training Started!")
    print(friendly_prompt() + "You can press Ctrl+C to interrupt training.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[red]Training interrupted by user. Saving the current model...[/red]")
        interrupted_model_path = Prompt.ask(friendly_prompt() + "Save interrupted model to?", default='./interrupted-model')
        model.save_pretrained(interrupted_model_path)
        tokenizer.save_pretrained(interrupted_model_path)
        print(friendly_prompt() + f"Model saved to '{interrupted_model_path}' after interruption.")
        return

    print(friendly_prompt() + "Training Finished!")
    finished_model_path = Prompt.ask(friendly_prompt() + "Where do you want to save your finished model?", default='./your-own-model')

    model.save_pretrained(finished_model_path)
    tokenizer.save_pretrained(finished_model_path)

    print(friendly_prompt() + f"Final Model has Saved to '{finished_model_path}'!")

if __name__ == "__main__":
    main()
