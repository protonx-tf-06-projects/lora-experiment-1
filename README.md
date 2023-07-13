# lora-experiment-1
Implementation of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). 


<p align="center">
    <img src='https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/1ae57bae-def9-4319-a6c6-8d2f8ed47165' width=400 class="center">
</p>

This project is a part of TF06 Course from ProtonX. We use LoRA techique to improve training Large Language Model (Vietnamese and English Dataset).

***Give us a star if this repo helpful to you.***

Slide about LoRA Explain (by Nguyen Bui Ngoc Han):

- [LoRA Slide](https://github.com/protonx-tf-06-projects/lora-experiment-1/blob/main/document/LoRA.pptx)
- [LoRA Notion](https://exuberant-puppy-021.notion.site/LoRA-Low-Rank-Adaptation-of-Large-Language-Models-b671e47f984f4c5e962f3176084ea819)


## What did we do in this project?
We built 4 model....


## I.  How to run our pretrained model?
You just download the ipybn file and run it on Google Colab or on your Jupyter Notebook.

## II.  How to add LoRA to finetuining your own model?
- Step 1: Load your model.

  For example you have model like this:
```python
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
modelName = "bigscience/bloomz-1b1" # Or whatever you want in HuggingFace
model = AutoModelForCausalLM.from_pretrained(modelName).to(device)
tokenizer = AutoTokenizer.from_pretrained(modelName)
```
  The *device* is your hardware support. You can set it automatically with this code:
```python
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

- Step 2: Prepare dataset for Training.

  For example you want to make a text-generating model for question-anwsering task, you will need a dataset that it have list of questions and anwsers.
  You can try this dataset for practice:
    -- [Kaggle Ecommerce FAQ Chatbot Dataset](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)
```

```

- Step 3: 

## II.  About datasets
In this project web
## IV. Result and Comparision

## . Result and Comparision


### **Authors:**
Nguyen Thanh Phat (phatjk)
- Github: https://github.com/phatjkk
- Linkedin: https://www.linkedin.com/in/phatjk
- Email: autoittutorial1234@gmail.com

Nguyen Bui Ngoc Han (Nguyễn Hân)
- Github: https://github.com/nbngochan
- Linkedin: https://www.linkedin.com/in/nbngochan99
- Email: nbngochan99@gmail.com

Nguyen Thanh Chung (Edward Nguyen)
- Github: https://github.com/dean6969
- Linkedin: https://www.linkedin.com/in/edwardngt
- Email: edwardngu96@gmail.com

Pham Quynh Trang (Trang Pham)
- Github: https://github.com/
- Linkedin: https://www.linkedin.com/in/
- Email: phamthiquynhtrang95@gmail.com
  
### **Advisors:**
Nguyen Ba Ngoc
- Github: https://github.com/bangoc123
- Linkedin: https://www.linkedin.com/in/nbangoc
- Email: protonxai@gmail.com
