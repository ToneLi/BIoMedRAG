# from datasets import load_dataset
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset



def formatting_func(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}

def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    # print( formatted_data["train"])
    return formatted_data["train"]



train_path="/home/li003378/project1/llama_project/0_Sunflower_triplet_extraction_13b/make_chuck_triplet_data/train_chuck_5_triplet_llama2_13b_right_2.json"
test_path="/home/li003378/project1/llama_project/0_Sunflower_triplet_extraction_13b/make_chuck_triplet_data/test_chuck_5_triplet_llama2_13b_right_2.json"
train = prepare_data(train_path)
test = prepare_data(test_path)

# print(formatted_dataset["train"][2]["text"])


# model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
model_id="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf"


qlora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto',
)



tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})



supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train,
    eval_dataset=test,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=10000,
        save_steps=1000,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        output_dir="/PETAILOR/TE_llama2_13b_ourmethod_chuck5_right_2",
        optim="paged_adamw_8bit",
        fp16=True,
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=512
)

supervised_finetuning_trainer.train()



"""
CUDA_VISIBLE_DEVICES=2  nohup  python 1_ourmethod_chuck5_sim_llama2_13b_right_0.py >myout.1_ourmethod_chuck5_sim_llama2_13b_right_0 2>&1 &   
CUDA_VISIBLE_DEVICES=3  nohup  python 1_ourmethod_chuck5_sim_llama2_13b_right_1.py >myout.1_ourmethod_chuck5_sim_llama2_13b_right_1 2>&1 &  
CUDA_VISIBLE_DEVICES=4  nohup  python 1_ourmethod_chuck5_sim_llama2_13b_right_2.py >myout.1_ourmethod_chuck5_sim_llama2_13b_right_2 2>&1 &  
"""