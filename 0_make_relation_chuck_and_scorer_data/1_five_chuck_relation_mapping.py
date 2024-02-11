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

def relation_with_defination():
    RD={}
    with open("relation_with_defination.txt","r") as fr:
        i=-1
        for line in fr.readlines():
            i=i+1
            line=line.strip().split("    ")
            RD[line[0]]=i
    return RD
            
R_D=relation_with_defination()

R_D_embedding_matrix=np.load("relation_description.npy")

def make_text_embeddings(discription):
    input_ids = tokenizer.encode(discription, return_tensors='pt')
    output = model(input_ids, output_hidden_states=True)  # <= set output_hidden_states to True
    hidden_states = output.hidden_states  # all the hidden_states are collected in this tuple
    input_embeddings = hidden_states[0]  # get the input embeddings
    # final_embeddings = hidden_states[-1][0][0].detach().numpy() # get the final embeddings
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)

    final_embeddings = input_embeddings.detach().numpy()

    return final_embeddings




def make_chunck(one_chuck_n_words):
    fw_=open("relation_with_chucks_5.txt","w")
    relation_chucks={}
    j=-1
    with open("renew_triplet_T_dev.jsonl","r") as fr:
        for line in fr.readlines():
            j=j+1
            print(j)
            line=json.loads(line)
            # print(line)
            relation=line["PREDICATE"]
          
            relation_embedding=R_D_embedding_matrix[R_D[relation]]
            # print(realtion_discription)
            sentence=line["SENTENCE"].split(" ")

            re_sentence=[]
            for s  in sentence:
                if len(s)!=0:
                    re_sentence.append(s)

            # print(re_sentence)
            chunk_size=len(re_sentence)/one_chuck_n_words
            # print(chunk_size)

            chunked_data = np.array_split(re_sentence, chunk_size)
            
            chucned_data_new=[]
            cos_=[]
            for chuck  in chunked_data:
                chuck=" ".join(chuck)
                chucned_data_new.append(chuck)
                chuck_embedding=make_text_embeddings(chuck)
        
                cos_.append(cos_similisty(relation_embedding,chuck_embedding))

            if len(cos_)>=2:
                top_2_idx = np.argsort(cos_)[-2:]
                top_2_values = [chucned_data_new[i] for i in top_2_idx]
                if relation not in relation_chucks:
                    relation_chucks[relation]=top_2_values
                else:
                    relation_chucks[relation]= relation_chucks[relation]+top_2_values
                
            else:
                if relation not in relation_chucks:
                    relation_chucks[relation]=chucned_data_new
                else:
                    relation_chucks[relation]= relation_chucks[relation]+chucned_data_new


    for relation, chucks in relation_chucks.items():
        DIc_={}
        DIc_[relation]=chucks
        fw_.write(json.dumps(DIc_))
        fw_.write("\n")

if __name__=="__main__":
    one_chuck_n_words=5#3,4,5
    make_chunck(one_chuck_n_words)
