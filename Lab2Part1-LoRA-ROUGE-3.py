# In[ ]:
import sys
import os
import site
from pathlib import Path

!echo "Installation in progress, please wait..."
!{sys.executable} -m pip cache purge > /dev/null

%pip install --user --upgrade transformers datasets trl peft accelerate scipy sentencepiece ipywidgets evaluate rouge_score --no-warn-script-location

!echo "Installation completed."

# Get the site-packages directory
site_packages_dir = site.getsitepackages()[0]

# add the site pkg directory where these pkgs are insalled to the top of sys.path
if not os.access(site_packages_dir, os.W_OK):
    user_site_packages_dir = site.getusersitepackages()
    if user_site_packages_dir in sys.path:
        sys.path.remove(user_site_packages_dir)
    sys.path.insert(0, user_site_packages_dir)
else:
    if site_packages_dir in sys.path:
        sys.path.remove(site_packages_dir)
    sys.path.insert(0, site_packages_dir)


# In[ ]:
import torch
import os

os.environ["WANDB_DISABLED"] = "true"
import transformers
from transformers import AutoTokenizer
from peft import LoraConfig
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from evaluate import load


# In[ ]:
from huggingface_hub import notebook_login
notebook_login()


# In[ ]:
model_path = "google/gemma-2-2b-it"

# TODO: create tokenizer using AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            attn_implementation='eager',
                                           device_map="auto")


# In[ ]:
def process_dataset(sample):
    messages = [
        {"role": "user", "content": f"Instruction:\nSummarize the following article.\n\nInput:\n{sample['Articles']}"},
        {"role": "assistant", "content": sample['Summaries']}
    ]
    sample = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
    return sample

dataset = load_dataset("gopalkalpande/bbc-news-summary", split="train")
dataset = dataset.map(process_dataset)

split_dataset = dataset.train_test_split(test_size=0.1, seed=99)
train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]


# In[ ]:
rouge = load('rouge')

# initialize lists of predictions and references later used to compute rouge scores
predictions = []
references = []

# iterate through the first 15 samples
for article, abstract in zip(validation_dataset["Articles"][:15], validation_dataset["Summaries"][:15]):
    messages = [
        {"role": "user", "content": f"Instruction:\nSummarize the following article.\n\nInput:\n{article}"},
    ]
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
    
    # TODO: perform model inference using the tokens in ``input_ids''
    output = model.generate(input_ids, max_new_tokens=150)
     
    # Remove input prompt from output
    prompt_length = input_ids.shape[1]
    answer = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    
    # TODO: add one answer to the ``predictions'' list, which is later passed to rouge compute
    predictions.append(answer) 
    # TODO: add one abstract to the ``references'' list, which is later passed to rouge compute
    references.append(abstract) 
    
    print(100*'-')
    print("Abstract:", abstract)
    print(100*'-')
    print("Model Summary:", answer)

print(100*'-')
# TODO: compute and print out the rouge scores including rouge1, rouge2, rougeL and rougeLsum
# TODO: you can refer to https://huggingface.co/spaces/evaluate-metric/rouge/blob/main/README.md#how-to-use
# print(...)

scores = rouge.compute(
    predictions=predictions,
    references=references,
    rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
)

print("ROUGE Scores:")
print("ROUGE-1:", scores["rouge1"])
print("ROUGE-2:", scores["rouge2"])
print("ROUGE-L:", scores["rougeL"])
print("ROUGE-Lsum:", scores["rougeLsum"])

print(100*'-')


# In[ ]:
finetuned_model = "gemma-2-2b-it-finetuned"

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# TODO: set up the trainer using SFTTrainer class
# TODO: you can refer to the gemma_xpu_finetuning.ipynb exercise
# TODO: this part is relatively long because of the arguments that need to be set
# trainer = SFTTrainer(...)
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    output_dir=f"./{finetuned_model}",
    bf16=True,
    max_grad_norm=0.3,
    weight_decay=0.01,
    max_length=16, 
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
result = trainer.train()
model.config.use_cache = True
print(result)


# In[ ]:
finetuned_model = "gemma-2-2b-it-finetuned"


rouge = load('rouge')

finetuned_model_path = f"{finetuned_model}/checkpoint-501"
#loaded_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, device_map="cuda")
loaded_model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_path,
    device_map={"": 0}
)
#tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
predictions = []
references = []

# TODO: compute rouge scores on the first 15 sample again.
# TODO: you can repeat the code from the earlier cells.
#
for article, abstract in zip(validation_dataset["Articles"][:15], validation_dataset["Summaries"][:15]):
    messages = [
        {"role": "user", "content": f"Instruction:\nSummarize the following article.\n\nInput:\n{article}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    output = loaded_model.generate(input_ids, max_new_tokens=150)

    prompt_length = input_ids.shape[1]
    answer = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

    predictions.append(answer)
    references.append(abstract)

scores = rouge.compute(
    predictions=predictions,
    references=references,
    rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
)

print("ROUGE Scores:")
print("ROUGE-1:", scores["rouge1"])
print("ROUGE-2:", scores["rouge2"])
print("ROUGE-L:", scores["rougeL"])
print("ROUGE-Lsum:", scores["rougeLsum"])


# In[ ]:
import torch

print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")


# In[ ]:
!pip install protobuf


# In[ ]:


