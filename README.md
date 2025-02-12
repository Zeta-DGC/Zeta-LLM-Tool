
![HINODE AI COM](https://github.com/user-attachments/assets/a1e37084-a2ab-48cb-a88e-3e125940f19f)

---

```
Documents/Hinode.ai/Normal via 🐍 v3.11.11 
❯ python execute.py
| /q to quit, /c to continue generation. |                                                                                                                                                             
>>>What is Xcode?
Xcode is a development environment for iOS and Android that allows developers to create and share apps quickly and easily.                                                                                                                                                                                                                                         
>>>Who made it?
It was developed by Apple and was the first major programming environment to include Swift programming language.                                                                                                                                                                                                                                               
>>>
```

# Hinode-AI
Fully Open-source LLM Tool.
- **training.py:** Learning using Azuki-Formatted Dataset
- **execute.py:** Run Learned Model (You Need Move Model Folder to ./trained-model)

## Built-in Dataset
- (Need Git-LFS to Clone This) **[OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)** by OpenO1 Team **I converted to Azuki-Formatted Dataset.**
- **[Azuki AI](https://github.com/DiamondGotCat/Azuki.ai) 2n** by Me(DiamondGotCat)

## Tokenizer
- **GPT2** (for OpenAI's Models)
- **Custom** (You Need Edit **Script** and **Base Model**)

## Selectable Base Models

### OpenAI
OpenAI's Latest Public Model.
We can't use GPT3.0 and later...

- **GPT2** Start from Scratch (Tokenizer: GPT2)
- **GPT2-Small** Start from Small Data (Tokenizer: GPT2)
- **GPT2-Medium** Start from Medium Data (Tokenizer: GPT2)

## Pre-trained Models (Hugging Face)
- [0.1](https://huggingface.co/DiamondGotCat/Hinode-AI-0.1): Using GPT-2 Tokenizer, GPT-2 Medium, Azuki AI 2n Dateset to Learning.

---

Thank you for reading this.

Actually, Hinode-AI is a personal project. There are still parts that haven't been developed yet.

If possible, please help in one of the following ways:
1. Simple: Please give it a star.
2. For programmers/engineers: Help with code fixes or testing.
3. For those who can support Hinode-AI's future: Publish the trained models on HuggingFace. However, please include information about the Hinode-AI project. For more details, see [Help with Trained Model](Help/Training.md).
