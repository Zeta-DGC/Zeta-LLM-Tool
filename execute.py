import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rich.console import Console
from rich.markdown import Markdown

model_save_path = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
model = GPT2LMHeadModel.from_pretrained(model_save_path)
console = Console()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = ""
def myPrint(content):
    console.print(Markdown(f"{content}"))

myPrint("| /q to quit, /c to continue generation. |")

while True:

    try:
        Input = input(">>>")

        if Input == "/q":
            break

        elif Input == "/c":
            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = generated_text[len(prompt):]
            prompt += result

            myPrint(f"{result}")

        else:
            prompt += "<|input|>" + Input + "<|end|>"
            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result1 = generated_text[len(prompt):].split("<|")[0]
            result2 = generated_text[len(prompt):].split("<|")[1]
            prompt += result1 + result2

            myPrint(f"{result1}")
            myPrint("---")
            myPrint(f"{result2}")

    except KeyboardInterrupt:
        break
