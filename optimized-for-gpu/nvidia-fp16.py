import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
from rich import print
from rich.prompt import Prompt
from datasets import load_dataset

print("Zeta-Tool Optimized Version: NVIDIA GPUs that Support float16")
print("""
[blue]--- RunPod.io (NVIDIA GPU) ---[/blue]
[bold]ALL[/bold]
""")

print("Zeta-Tool Optimized Version: NVIDIA GPUs that Support bfloat16")
print("""
[blue]--- RunPod.io (NVIDIA GPU) ---[/blue]
[bold]32GB VRAM AND UNDER:[/bold] L4, A30
[bold]48GB VRAM:[/bold] RTX 6000 Ada, L40S, L40, A40
[bold]80GB VRAM:[/bold] A100 SXM, A100 PCIe, H100 SXM, H100 PCIe
[bold]80GB+ VRAM:[/bold] H200 SXM, H100 NVL, NVIDIA B200
""")

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

    print("Zeta-Tool")
    print("    You can Create LLM from Other LLM")
    print("----------")
    print("")

    friendly_name = {}
    friendly_name["maker"] = "(Default)"
    friendly_name["collection"] = "(Default)"
    friendly_name["model"] = "(Default)"

    def friendly_prompt():
        return f"Zeta-Tool> {friendly_name['maker']}> {friendly_name['collection']}> {friendly_name['model']}> "

    maker = Prompt.ask("Zeta-Tool> Maker ID", choices=["zeta", "openai", "google", "meta", "alibaba", "local", "custom"], default="zeta")

    if maker == "zeta":

        friendly_name["maker"] = "Zeta Project"
        model_id = Prompt.ask("Zeta-Tool> Zeta Project> Model ID", choices=["zeta-1", "zeta-2", "zeta-3", "zeta-4"], default="zeta-4")

        if model_id == "zeta-1":
            friendly_name["model"] = "Zeta 1"
            tokenizer = AutoTokenizer.from_pretrained('Zeta-LLM/Zeta-1')
            model = AutoModelForCausalLM.from_pretrained('Zeta-LLM/Zeta-1')

        if model_id == "zeta-2":
            friendly_name["model"] = "Zeta 2"
            tokenizer = AutoTokenizer.from_pretrained('Zeta-LLM/Zeta-2')
            model = AutoModelForCausalLM.from_pretrained('Zeta-LLM/Zeta-2')

        if model_id == "zeta-3":
            friendly_name["model"] = "Zeta 3"
            tokenizer = AutoTokenizer.from_pretrained('Zeta-LLM/Zeta-3')
            model = AutoModelForCausalLM.from_pretrained('Zeta-LLM/Zeta-3')

        if model_id == "zeta-4":
            friendly_name["model"] = "Zeta 4"
            tokenizer = AutoTokenizer.from_pretrained('Zeta-LLM/Zeta-4')
            model = AutoModelForCausalLM.from_pretrained('Zeta-LLM/Zeta-4')

    elif maker == "openai":

        friendly_name["maker"] = "OpenAI"
        model_id = Prompt.ask("Zeta-Tool> OpenAI> Model ID (Select gpt2 to Making from Scratch)", choices=["gpt2", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2-small")

        if model_id == "gpt2":
            friendly_name["model"] = "GPT-2 (Tokenizer Only)"
            config = GPT2Config()
            model = GPT2LMHeadModel(config)
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        elif model_id == "gpt2-small":
            friendly_name["model"] = "GPT-2 (Small)"
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
        elif model_id == "gpt2-medium":
            friendly_name["model"] = "GPT-2 (Medium)"
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-medium')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-medium')
        elif model_id == "gpt2-large":
            friendly_name["model"] = "GPT-2 (Large)"
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large')
        elif model_id == "gpt2-xl":
            friendly_name["model"] = "GPT-2 (X-Large)"
            tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
            model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-xl')

    elif maker == "google":

        friendly_name["maker"] = "Google Deepmind"

        model_id = Prompt.ask("Zeta-Tool> Google> Model ID (Permission Required)", choices=["gemma", "codegemma", "gemma1.1", "gemma2"], default="gemma")

        if model_id == "gemma":

            friendly_name["collection"] = "Gemma 1"

            size = Prompt.ask("Zeta-Tool> Google> Gemma 1> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":

                friendly_name["model"] = "2b"

                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')

            elif size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-7b')

        elif model_id == "codegemma":

            friendly_name["collection"] = "CodeGemma 1"

            size = Prompt.ask("Zeta-Tool> Google> CodeGemma 1> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":

                friendly_name["model"] = "2b"

                tokenizer = AutoTokenizer.from_pretrained('google/codegemma-2b')
                model = AutoModelForCausalLM.from_pretrained('google/codegemma-2b')

            elif size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('google/codegemma-7b')
                model = AutoModelForCausalLM.from_pretrained('google/codegemma-7b')

        elif model_id == "gemma1.1":

            friendly_name["collection"] = "Gemma 1.1 [Instruct]"

            size = Prompt.ask("Zeta-Tool> Google> Gemma 1.1 [Instruct]> Size", choices=["2b", "7b"], default="2b")

            if size == "2b":

                friendly_name["model"] = "2b"

                tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-2b-it')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-1.1-2b-it')

            elif size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('google/gemma-1.1-7b-it')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-1.1-7b-it')

        elif model_id == "gemma2":

            friendly_name["collection"] = "Gemma 2"
            size = Prompt.ask("Zeta-Tool> Google> Gemma 2> Size", choices=["2b", "9b", "27b"], default="2b")

            if size == "2b":

                friendly_name["model"] = "2b"

                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

            elif size == "9b":

                friendly_name["model"] = "9b"
                
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-9b')

            elif size == "27b":

                friendly_name["model"] = "27b"
                
                tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-27b')
                model = AutoModelForCausalLM.from_pretrained('google/gemma-2-27b')


    elif maker == "meta":

        friendly_name["maker"] = "Meta"

        model_id = Prompt.ask("Zeta-Tool> Meta> Model ID (Permission Required)", choices=["llama2", "codellama", "llama3", "llama3.1", "llama3.2", "llama3.2-v", "llama3.3"], default="llama2")

        if model_id == "llama2":

            friendly_name["collection"] = "Llama 2"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 2> Size", choices=["7b", "13b", "70b"], default="7b")

            if size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b')

            elif size == "13b":

                friendly_name["model"] = "13b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b')

            elif size == "70b":

                friendly_name["model"] = "70b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b')

        elif model_id == "codellama":

            friendly_name["collection"] = "CodeLlama 1 [Hugging Face Format]"

            size = Prompt.ask("Zeta-Tool> Meta> CodeLlama 1 [Hugging Face Format]> Size", choices=["7b", "13b", "34b", "70b"], default="7b")

            if size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-7b-hf')

            elif size == "13b":

                friendly_name["model"] = "13b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-13b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-13b-hf')

            elif size == "34b":

                friendly_name["model"] = "34b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-34b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-34b-hf')

            elif size == "70b":

                friendly_name["model"] = "70b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-70b-hf')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/CodeLlama-70b-hf')

        elif model_id == "llama3":

            friendly_name["collection"] = "Llama 3"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 3> Size", choices=["8b", "70b"], default="8b")

            if size == "8b":

                friendly_name["model"] = "8b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')

            elif size == "70b":

                friendly_name["model"] = "70b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-70B')

        elif model_id == "llama3.1":

            friendly_name["collection"] = "Llama 3.1"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 3.1> Size", choices=["8b", "70b", "405b"], default="8b")

            if size == "8b":

                friendly_name["model"] = "8b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')

            elif size == "70b":

                friendly_name["model"] = "70b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-70B')

            elif size == "405b":

                friendly_name["model"] = "405b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-405B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-405B')

        elif model_id == "llama3.2":

            friendly_name["collection"] = "Llama 3.2"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 3.2> Size", choices=["1b", "3b"], default="1b")

            if size == "1b":

                friendly_name["model"] = "1b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

            elif size == "3b":

                friendly_name["model"] = "3b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')

        elif model_id == "llama3.2-v":

            friendly_name["collection"] = "Llama 3.2 [Vision]"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 3.2 [Vision]> Size", choices=["11b", "90b"], default="11b")

            if size == "11b":

                friendly_name["model"] = "11b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-11B-Vision')

            elif size == "90b":

                friendly_name["model"] = "90b"

                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-90B-Vision')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-90B-Vision')

        elif model_id == "llama3.3":

            friendly_name["collection"] = "Llama 3.3"

            size = Prompt.ask("Zeta-Tool> Meta> Llama 3.3> 70B", choices=["confirm"], default="confirm")

            if size == "confirm":
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')


    elif maker == "alibaba":

        friendly_name["maker"] = "Alibaba Cloud"

        model_id = Prompt.ask("Zeta-Tool> Alibaba Cloud> Model ID", choices=["qwen", "qwen1.5"], default="qwen")

        if model_id == "qwen":

            friendly_name["collection"] = "Qwen 1"

            size = Prompt.ask("Zeta-Tool> Alibaba Cloud> Qwen 1> Size", choices=["1.8b", "7b", "14b", "72b"], default="1.8b")

            if size == "1.8b":

                friendly_name["model"] = "1.8b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-1_8B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-1_8B')

            elif size == "7b":

                friendly_name["model"] = "7b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B')

            elif size == "14b":

                friendly_name["model"] = "14b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-14B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-14B')

            elif size == "72b":

                friendly_name["model"] = "72b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-72B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-72B')

        elif model_id == "qwen1.5":

            friendly_name["collection"] = "Qwen 1.5"

            size = Prompt.ask("Zeta-Tool> Alibaba Cloud> Qwen 1.5> Size", choices=["0.5b", "1.8b", "4b", "14b"], default="0.5b")

            if size == "0.5b":

                friendly_name["model"] = "0.5b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B')

            elif size == "1.8b":

                friendly_name["model"] = "1.8b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-1.8B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-1.8B')

            elif size == "4b":

                friendly_name["model"] = "4b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-4B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-4B')

            elif size == "14b":

                friendly_name["model"] = "14b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-14B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-14B')

            elif size == "32b":

                friendly_name["model"] = "32b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-32B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-32B')

            elif size == "72b":

                friendly_name["model"] = "72b"

                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-72B')
                model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-72B')

    elif maker == "local":

        friendly_name["maker"] = "Local"

        print("[yellow]WARNING:[/yellow] In Local Mode, Please enter Path of the Directory that Transformers model with full weights, not a quantized version such as GGUF.")
        trained_model_folder = Prompt.ask("Zeta-Tool> Local> Folder Path", default="./trained_model")

        friendly_name["model"] = trained_model_folder

        tokenizer = AutoTokenizer.from_pretrained(trained_model_folder)
        model = AutoModelForCausalLM.from_pretrained(trained_model_folder)

    elif maker == "custom":

        friendly_name["maker"] = "Custom"

        print("[yellow]WARNING:[/yellow] In Custom Mode, Please enter Path of the Repository on Hugging Face that Transformers model with full weights, not a quantized version such as GGUF.")
        hugging_face_repo = Prompt.ask("Zeta-Tool> Custom> Hugging Face Repository Path", default="openai-community/gpt2")

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
        print("[cyan]Available columns:[/cyan]", list(raw_dataset["train"].features.keys()))
        column_map["input"] = Prompt.ask(friendly_prompt() + "Which column contains input text?", default="input")
        column_map["output"] = Prompt.ask(friendly_prompt() + "Which column contains output text?", default="output")

        conversations = [
            f"<user>{ex[column_map['input']]}</user><assistant>{ex[column_map['output']]}</assistant>"
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
                role = str(message['role'])
                content = str(message['content'])
                convo_text += f"<{role.lower()}>{content}</{role.lower()}>"
            conversations.append(convo_text)

    df = pd.DataFrame({'conversation': conversations})

    train_dataset = ConversationDataset(df, tokenizer)

    training_args = TrainingArguments(
        output_dir=Prompt.ask(friendly_prompt() + "output_dir (Temporary)", default='./results'),
        num_train_epochs=Prompt.ask(friendly_prompt() + "num_train_epochs", default=3),
        per_device_train_batch_size=Prompt.ask(friendly_prompt() + "per_device_train_batch_size", default=1),
        gradient_accumulation_steps=Prompt.ask(friendly_prompt() + "gradient_accumulation_steps", default=4),
        learning_rate=Prompt.ask(friendly_prompt() + "learning_rate", default=3e-5),
        warmup_steps=Prompt.ask(friendly_prompt() + "warmup_steps", default=100),
        weight_decay=Prompt.ask(friendly_prompt() + "weight_decay", default=0.01),
        logging_dir=Prompt.ask(friendly_prompt() + "logging_dir", default='./logs'),
        logging_steps=Prompt.ask(friendly_prompt() + "logging_steps", default=10),
        fp16=True,
        save_strategy="no", # Avoid errors that occur during autosave
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

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
