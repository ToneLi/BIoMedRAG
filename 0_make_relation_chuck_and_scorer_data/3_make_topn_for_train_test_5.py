import json
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
import numpy as np
model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
import json

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


def cos_similisty(vector1,vector2):
    import numpy as np

    vector1 = np.array(vector1)

    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)

    norm_vector1 = np.linalg.norm(vector1)

    norm_vector2 = np.linalg.norm(vector2)

    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    return cosine_similarity



def make_text_embeddings(discription):
    input_ids = tokenizer.encode(discription, return_tensors='pt')
    output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
    hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
    input_embeddings = hidden_states[0]  # get the input embeddings
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)

    final_embeddings = input_embeddings.detach().numpy()
    return final_embeddings

def load_data_base_relation_chuck():
    Chucks=[]
    relations=[]
    with open("stored_chucks_with_relation_5.txt","r") as fr:
        for line in fr.readlines():
            line=line.strip().split("--##--")
            Chucks.append(line[0])
            relations.append(line[1])

    return Chucks,relations

relation_chunks,relation_chunks_with_r=load_data_base_relation_chuck()
relation_chunks_embeddings=np.load("retrived_chucks_5.npy")


def make_train_test_chunck(one_chuck_n_words):
    fw_=open("train_data_topn_chucks_5.json","w")

    relation_chucks={}
    j=-1
    all_l=[]
    with open("train_data_chucks_5.json","r") as fr:
        for line in fr.readlines():
            j=j+1
            print(j)
            line=json.loads(line)

            sentence=line["sentence"].split(" ")
            all_chucks=line["all_chucks"]  # one sentence many chucks
            # print(all_chucks)
            all_c_top_n={}
            for chucks in all_chucks:
               
                for ch ,itssimchuk in chucks.items():
                    # print("mmm",itssimchuk)
                    # print("ch",ch)
                    for i in range(len(itssimchuk[0][:1])):
                        all_c_top_n[itssimchuk[0][i]]=itssimchuk[1][i]

            line["topn_sim_chuck"]=all_c_top_n
            fw_.write(json.dumps(line))
            fw_.flush()
            fw_.write("\n")



if __name__=="__main__":
    one_chuck_n_words=5
    make_train_test_chunck(one_chuck_n_words)
