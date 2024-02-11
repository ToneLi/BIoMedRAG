
import json
from itertools import permutations
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

def sentence_chucks_mapping():
    DIc_={}
    j=-1
    with open("test_data_topn_chucks_5.json","r") as fr:
        for line in fr.readlines():
            j=j+1
            # print(j)
            line=json.loads(line)
            topn_sim_chuck=line["topn_sim_chuck"]
            all_lists_=[]
            for key, value in topn_sim_chuck.items():
                all_="Context: "+ key+" Response: "+ value
                all_lists_.append(all_)
            chucks_=make_chucks(all_lists_)
            # print(chucks_)
            all_lists_chucks=[]
            for c in chucks_:
                if len(c) !=0:
                    c=" ".join(c)
                    all_lists_chucks.append(c)
            DIc_[j]=all_lists_chucks
    return DIc_
            
            
SC_map=sentence_chucks_mapping()


"""
entity_dict = "/home/li003378/project1/llama_project/5_make_chuck/all_candiate_chunks_5.txt"
entity_path = "/home/li003378/project1/llama_project/5_make_chuck/all_posible_chuck_vector_5_llama2_13b.npy"

"""
def redefine_instruction():
    fw=open("test_chuck_instruction_5_llama13b_right.json","w")

    j=-1
    with open("KNN_demo_test_RE.json","r") as fr:
        # j=j+1  wrong operation
        for line in fr.readlines():
            j=j+1
            line=json.loads(line.strip())
            instruction=line["instruction"]
            chuck_isntruction="|".join(SC_map[j])
            # print(chuck_isntruction)
            instruction=instruction+"|"+chuck_isntruction
            line["instruction"]=instruction
            # print(instruction)
            fw.write(json.dumps(line))
            fw.write("\n")


redefine_instruction()