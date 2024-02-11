import json
import numpy as np
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
    # final_embeddings = hidden_states[-1][0][0].detach().numpy() # get the final embeddings
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
    fw_=open("train_data_chucks_5.json","w")

    relation_chucks={}
    j=-1
    with open("renew_triplet_T_train.jsonl","r") as fr:
        for line in fr.readlines():
            j=j+1
          
            if j>=0:
                print(j)
                line=json.loads(line)

                sentence=line["SENTENCE"].split(" ")

                re_sentence=[]
                for s  in sentence:
                    if len(s)!=0:
                        re_sentence.append(s)

                # print(re_sentence)
                if len(re_sentence) >one_chuck_n_words:
                    chunk_size=len(re_sentence)/one_chuck_n_words
                    # print(chunk_size)

                    chunked_data = np.array_split(re_sentence, chunk_size)
                    # print("chunked_data",chunked_data)
                else:
                    chunked_data=[re_sentence]
                    
                chucned_data_new=[]
                all_chucks=[]
                for chuck  in chunked_data:
                    cos_=[]
                    relations=[]

                    chuck=" ".join(chuck)
                    chucned_data_new.append(chuck)
                    chuck_embedding=make_text_embeddings(chuck)
                    i=-1
                    for cu in relation_chunks:
                        i=i+1
                        store_chuck_vector=relation_chunks_embeddings[i]
                        cos_.append(cos_similisty(chuck_embedding,store_chuck_vector))
                    
    
                    top_5_idx = np.argsort(cos_)[-5:]

                    top_5_chucks = [relation_chunks[i] for i in top_5_idx]
                    top_5_relations = [relation_chunks_with_r[i] for i in top_5_idx]
                    
                    chuck_sim_chucks={}
                    chuck_sim_chucks[chuck]=[top_5_chucks,top_5_relations]
                    all_chucks.append(chuck_sim_chucks)
                # print(all_chucks)
                final_dic={}
                final_dic["relation"]=line["PREDICATE"]
                final_dic["head"]=line["SUBJECT_TEXT"]
                final_dic["tail"]=line["OBJECT_TEXT"]
                final_dic["sentence"]=" ".join(re_sentence)
                final_dic["all_chucks"]=all_chucks
                fw_.write(json.dumps(final_dic))
                fw_.flush()
                fw_.write("\n")
                
                
                    
               
if __name__=="__main__":
    one_chuck_n_words=5
    make_train_test_chunck(one_chuck_n_words)
