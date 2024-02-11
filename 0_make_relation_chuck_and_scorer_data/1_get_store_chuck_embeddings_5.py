from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
import numpy as np
model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
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




re=[]
chucks=[]
relations=[]
i=0
with open("relation_with_chucks_5.json","r") as fr:
        for line in fr.readlines():
            line=json.loads(line)
            # i=i+1
            # print(i)
            for r, c in line.items():
                for chuck in list(set(c)):
                    i=i+1
                    print(i)
                    chucks.append(chuck)
                    relations.append(r)
            
                    input_ids = tokenizer.encode(chuck, return_tensors='pt')
                    output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
                    hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
                    input_embeddings = hidden_states[0]  # get the input embeddings
                    # final_embeddings = hidden_states[-1][0][0].detach().numpy() # get the final embeddings
                    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
                    final_embeddings = input_embeddings.detach().numpy()
                    re.append(final_embeddings)


np.save("retrived_chucks_5.npy", re)

fw=open("stored_chucks_with_relation_5.txt","w")

for i in range(len(chucks)):
    fw.write(chucks[i]+"--##--"+relations[i]+"\n")


"""
4-chucks
967

"""