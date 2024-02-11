import json
from itertools import permutations
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
import numpy as np
# model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
model_id="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf"

import json
# model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto',
)


tokenizer = LlamaTokenizer.from_pretrained(model_id)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def make_chucks(items):
    all_chucks=[]
    for i in range(len(items)+1):
        chucks=[]
        for p in permutations(items,i):
            chucks.append(p)
        if i==len(items):
            all_chucks=all_chucks+[chucks[0]]
        else:
            all_chucks = all_chucks +chucks
    return all_chucks



def get_chucks(file):
    j=-1
  
    all_c=[]
    with open(file,"r") as fr:
        for line in fr.readlines():
            j=j+1
            # print(j)
            line=json.loads(line)
            topn_sim_chuck=line["topn_sim_chuck"]
            # print(topn_sim_chuck)
            all_lists_=[]
            for key, value in topn_sim_chuck.items():
                all_="Context: "+ key+" Response: "+ value
                all_lists_.append(all_)
            chucks_=make_chucks(all_lists_)
            # print(chucks_)
            for c in chucks_:
                if len(c) !=0:
                    c=" ".join(c)
                    all_c.append(c)
    return all_c




def get_topn_whole_sentence(file):
    j=-1
  
    all_c=[]
    with open(file,"r") as fr:
        for line in fr.readlines():
            j=j+1
            # print(j)
            line=json.loads(line)
            context=line["context"]
            instruction=line["instruction"].split("Examples:  ")[1]
            # print(instruction)
            all_c.append(instruction)
            all_c.append(context)


    return all_c




fw=open("all_candiate_chunks_5.txt","w")
train_=get_chucks("train_data_topn_chucks_5.json")
test_=get_chucks("test_data_topn_chucks_5.json")
print("---------train_----finished---")

train_topn=get_topn_whole_sentence("KNN_demo_train_RE.json")
test_topn=get_topn_whole_sentence("KNN_demo_test_RE.json")

all_chuck_vectors=[]
flg=0
for chg in set(train_+test_+train_topn+test_topn):
    flg=flg+1
    print(flg)
    fw.write(chg)
    fw.write("\n")
    fw.flush()

    input_ids = tokenizer.encode(chg, return_tensors='pt')
    output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
    hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
    input_embeddings = hidden_states[0]  # get the input embeddings
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)

    final_embeddings = input_embeddings.detach().numpy()
    all_chuck_vectors.append(final_embeddings)

np.save("all_posible_chuck_vector_5_llama2_13b.npy", all_chuck_vectors)

"""
top2 137730
top1: 81
"""