
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, torch_dtype=torch.float16, device_map='auto',
)

tokenizer = LlamaTokenizer.from_pretrained(model_id)


all_vectors=[]
with open("relation_with_defination.txt", "r") as fr:
    i = -1
    for line in fr.readlines():
        i = i + 1
        line = line.strip().split("\t")[1]

        input_ids = tokenizer.encode(line, return_tensors='pt')
        output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
        hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
        input_embeddings = hidden_states[0]  # get the input embeddings
        input_embeddings=torch.mean(input_embeddings.squeeze(0),dim=0)

        final_embeddings=input_embeddings.detach().numpy()
        all_vectors.append(final_embeddings)



