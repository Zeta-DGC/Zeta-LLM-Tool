Thank you for your interest in this project.  
This document explains one way to support Zeta-Tool: contributing via pre-trained models.

### Requirements

Before proceeding, ensure you have the following:  
1. Time  
2. Electricity costs  
3. A basic understanding of Python  
4. A relatively powerful machine  
5. Familiarity with Hugging Face (for publishing)  

This approach requires a machine with at least **16GB of GPU memory (VRAM)**.  
The weaker your GPU, the longer the process will take.  

For reference, training using the Azuki-2n dataset on an M3 chip with 16GB of shared RAM takes at least 30 minutes and can take up to 2 hours.  
If you use the same machine for other tasks during training, it may take even longer.  
Thus, using a secondary PC is recommended.

---

### 1. Ensure Python and Dependencies Is Installed

If Python is not installed, the easiest method is to use **Anaconda** or **Miniconda**.  
For advanced users, **Pyenv** can also be used on Linux or macOS.  

We recommend Python versions **3.10** or **3.11** for stability.

If Dependencies of This Project are not Installed, Please Install.
Example:
```
pip3 install pandas transformers torch rich
```

---

### 2. Clone the GitHub Repository

Clone this repository using the **Git command**, **GitHub CLI**, or a GUI tool.

---

### 3. Training the Model

The most time-consuming step is training the model.  
To make this easier, a `training.py` script is located in the root directory.

💡 Note: If you do not have access to a GPU, it is still possible to fine-tune small models using CPU-only machines. However, this is significantly slower and may require reducing model size and batch size.

#### Training Options

After launching `training.py`, you will be prompted to make a selection.

#### Dataset Path

Next, you will be asked to provide the path to the dataset. Zeta-Tool uses a custom JSON format.  
We recommend selecting one of the provided templates.

Zeta-Tool includes the following datasets as of this writing:  
- (Need Git-LFS to Clone This) **[OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)** by OpenO1 Team **I converted to Azuki-Formatted Dataset.**
- **Zeta-Classic 2n** by DiamondGotCat

Here is a table summarizing the characteristics of the included datasets:

| Dataset Name | Characteristics | Base Model | Training Time | Notes | Path |
|--------------|-----------------|------------|----------------|-------|------|
| **OpenO1** | Published by the OpenO1 team on Hugging Face.<br> Includes reasoning processes for better answers.<br> High information density. | openai-community/gpt2 (GPT-2 Small) | Long (~12 hours) | May yield high-quality answers.<br> Training time tends to be long due to high data volume. | `data_templates/OpenO1-SFT.json` |
| **Azuki 2n** | Created for version 2n of the Azuki.ai project.<br> Lower answer quality than Zeta-Tool. | openai-community/gpt2-large (GPT-2 Large) | Moderate (~30 min–2 hrs)  | May lack information density, so v2-medium or higher is recommended. | `data_templates/azuki-2n.json` |

---

### 4. Outputs

Intermediate results will be saved in the `results` folder.  
The final model will be saved in the `trained_model` folder.

---

### 5. Test Your Model (Optional)

Once training is complete, use `execute.py` to interact with your model.

```bash
python3 execute.py
```

### 6. Upload Your Model

Finally, upload the contents of the `trained_model` folder to your own account as a model.  
Include the following details in the description:  
- Information about Zeta-Tool  
- The base model used  
- The name of the dataset used

💡 Example Model Name: `azuki-2n-gpt2-large-remix`
