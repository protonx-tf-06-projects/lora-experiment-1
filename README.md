# lora-experiment-1
Implementation of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). 


<p align="center">
    <img src='https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/1ae57bae-def9-4319-a6c6-8d2f8ed47165' width=400 class="center">
</p>

This project is a part of TF06 Course from ProtonX. We use LoRA techique to improve training Large Language Model.

We use [Bloomz-1b1](https://huggingface.co/bigscience/bloomz-1b1) to fine tuning on English - Vietnamese datasets.

***Give us a star if this repo helpful to you.***

Slide about LoRA Explain (by Nguyen Bui Ngoc Han):

- [LoRA Slide](https://github.com/protonx-tf-06-projects/lora-experiment-1/blob/main/document/LoRA.pptx)
- [LoRA Notion](https://exuberant-puppy-021.notion.site/LoRA-Low-Rank-Adaptation-of-Large-Language-Models-b671e47f984f4c5e962f3176084ea819)


## I.  How to run our pretrained model?
You just download the .ipybn file and run it on Google Colab or on your Jupyter Notebook.

![image](https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/3b0dfd49-18a7-4a9b-aaab-64ccde0a70f0)


Live demo (Click icon below to run in Colab):

<a href="https://colab.research.google.com/drive/1tO13UP15_32JYBD7wSyAhUqztWHjWnkc?usp=sharing"><img src="https://storage.googleapis.com/protonx-cloud-storage/colab_favicon_256px.png" width=80> </a>


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

  + [Kaggle Ecommerce FAQ Chatbot Dataset](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)

  Get dataset from source: 
  ```
    !wget https://raw.githubusercontent.com/phatjkk/data/main/LLM/Ecommerce_FAQ_Chatbot_dataset.json
  ```
  Load dataset as HuggingFace Dataset type:
  ```python
    from datasets import load_dataset
    from datasets import Dataset
    data = load_dataset('json', data_files='Ecommerce_FAQ_Chatbot_dataset.json')
    ds = Dataset.from_list(data["train"]["questions"][0])
  ```
  Merge *question* and *answer* columns into one call *prediction*:
  ```python
    def merge_columns(example):
        example["prediction"] = example["question"] + " ->: " + str(example["answer"])
        return example
    # Map merge_columns function to dataset
    ds = ds.map(merge_columns)
  ```
  Tokenizer *prediction* column:
  ```python
    # Tokenizer/Véc tơ hóa văn bản (Chuyển văn bản thành số để training)
    def tokeni(example):
        example["prediction_token"] = tokenizer(example["prediction"], return_tensors='pt', padding=True)['input_ids']
        return example
    # Map tokeni function to dataset
    ds = ds.map(tokeni,batched=True)
  ```
- Step 3: Add LoraConfig Adapter to model
  ```python
    # Set config for LoRA 
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=16, #attention heads
        lora_alpha=16, #alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )
    # Set peft adapter to model
    model_lora = get_peft_model(model, config)
  ```
  There are some explain arguments for this code:
    - `r`: Lora attention dimension (int).
    - `lora_alpha`: The alpha parameter for Lora scaling.
    - `lora_dropout`: The dropout probability for Lora layers.
    - `bias`: Bias type for Lora. Can be 'none', 'all' or 'lora_only'
    - `task_type`: Task you want to run
- Step 4: Training model
  ```python
    # Training model
    import transformers
    from transformers import Trainer,EarlyStoppingCallback
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            #Perplexity
            perplexity = torch.exp(outputs.loss)
            return (perplexity, outputs) if return_outputs else perplexity
    trainer = CustomTrainer(
        model=model,
        train_dataset=ds_tt["train"]["prediction"],
        eval_dataset=ds_tt["test"]["prediction"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=3, # batch size
            num_train_epochs=1, # epochs
            gradient_accumulation_steps=1,
            warmup_steps=100,
            save_total_limit=5,
            learning_rate=2e-4,
            fp16=True,
            output_dir='outputs',
            logging_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end = True
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience = 4)]
    )
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
    trainer.train()
  ```

  When finish training task you can show the loss curve of train and validation:

   ```python
    trainingEpoch_loss_adam,validationEpoch_loss_adam=[],[]
    t = 0
    for i in trainer.state.log_history[:-1]:
      if t == 0:
        trainingEpoch_loss_adam.append(i["loss"])
        t=1
      else:
        validationEpoch_loss_adam.append(i["eval_loss"])
        t=0
    from matplotlib import pyplot as plt
    plt.plot(trainingEpoch_loss_adam, label='train_loss')
    plt.plot(validationEpoch_loss_adam,label='val_loss')
    plt.legend()
    plt.show
   ```

   Example result:
<p align="center">
    <img src='https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/8a8e0143-6013-48f5-83f7-ac71f8dbd0e6' width=700 class="center">
</p>
- Step 5: Test generate task

  You can gennerate text from model like this:

  ```python
    question = "How can I create an account?"
    prompt = question+" ->: "
    inputs = tokenizer( question, return_tensors="pt")
    with torch.autocast(device.type):
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=100)
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
  ```

  Example Result:

  ```
  How can I create an account? ->:  Click the "Create an account" button. Enter your email address and password. Click the "Continue" button.
  ```
  
## II.  About datasets
In this project we use data set from 3 source:
+ [Kaggle Ecommerce FAQ Chatbot Dataset (English)](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)
+ [Kaggle Ecommerce FAQ Chatbot Dataset (Vietnamese)](https://github.com/phatjkk/data/blob/main/LLM/Ecommerce_FAQ_Chatbot_dataset_vi.xlsx)
+ [UIT-ViQuAD](https://paperswithcode.com/dataset/uit-viquad)
+ [NLLB_translations_Vietnamese_51k](https://github.com/phatjkk/Vietnamese_LLMs/tree/main/Generate_and_Translate_Dataset/Vietnamese_Instructions_datasets/Translation/Alpaca_52k/NLLB_1B3_results)

## IV. Result and Comparision

Model result:

    - NLLB + viquad Dataset (Vietnamese): (training_loss=2.1773)
    - Ecommerce FAQ Chatbot Dataset (English): (training_loss=2.3110)
    - Ecommerce FAQ Chatbot Dataset (Vietnamese): (training_loss=2.0299)

Time compare:

+ Model bloomz-1b1 train data NLLB, 1 epoch (Using LoRA) (Train on V100 Colab)
<p align="center">
    <img src='https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/b9792b09-bd1f-4455-ad6e-6ad89c54ddb3' width=700 class="center">
</p>

+ Model bloomz-1b1 train data NLLB, 1 epoch (without LoRA) (Train on V100 Colab)
<p align="center">
    <img src='https://github.com/protonx-tf-06-projects/lora-experiment-1/assets/48487157/88da04cb-3ed5-4e3d-b6ee-00c7afe534bc' width=700 class="center">
</p>

Compare Table: 

|                | LoRA  | Without LoRA |
|----------------|-------|--------------|
| Time  Training | ~157m | ~202m        |

So with LoRA technique, we reduce the training time **22.2%** in NLLB-57k dataset with bloomz-1b1 model.

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
- Email: phamthiquynhtrang95@gmail.com
  
### **Advisors:**
Nguyen Ba Ngoc
- Github: https://github.com/bangoc123
- Linkedin: https://www.linkedin.com/in/nbangoc
- Email: protonxai@gmail.com
