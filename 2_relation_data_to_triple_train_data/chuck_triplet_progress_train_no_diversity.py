import json
import random
# relation_chuck_data_train="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/train_chuck_final_ICL_t2.json"
# relation_chuck_data_test="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/test_chuck_final_ICL_t2.json"

# relation_chuck_data_train="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/train_chuck5_s_r_sim_ablation.json"
# relation_chuck_data_test="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/test_chuck5_s_r_sim_ablation.json"


# relation_chuck_data_train="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/train_chuck3_s_r_sim.json"
# relation_chuck_data_test="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/0_models/data/test_chuck3_s_r_random.json"

relation_chuck_data_train="/home/li003378/project1/llama_project/0_Final_code_triplet/0_train_retrival_5_llama2_13b_TE_and_RE_2/data/train_chuck5_s_r_llama2_13b_right_one_no_diversity.json"
relation_chuck_data_test="/home/li003378/project1/llama_project/0_Final_code_triplet/0_train_retrival_5_llama2_13b_TE_and_RE_2/data/test_chuck5_s_r_llama2_13b_right_one_no_diversity.json"


triplet_KNN_top1_train="/home/li003378/project1/llama_project/0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test="/home/li003378/project1/llama_project/0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

dev_source="/home/li003378/project1/llama_project/1_Sunflower_Relation_extraction_13b/5_make_chuck/renew_triplet_T_dev.jsonl"

def HT_R():
    dic_={}
    with open(dev_source) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())
            h=line["SUBJECT_TEXT"]
            t=line["OBJECT_TEXT"]
            r=line["PREDICATE"]
            if r not in dic_:
                dic_[r]=[h+"|"+t]
            else:
                dic_[r]=dic_[r]+[h+"|"+t]
    return dic_

R_HT_mapping=HT_R()


def Relation_Chuck_train():
    dic_=[]
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
            # print(line)
            # break
            
    return dic_

RCTrain_=Relation_Chuck_train()




i=-1
j=-1

fw=open("train_chuck_5_triplet_llama2_13b_right_0_no_diversity.json","w")
with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            line=json.loads(line.strip())
            line_=line["instruction"].split("Examples:")
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
            instruction_=line_[0].strip()
            ex=RCTrain_[i]
            # print(excamples)

            sentence_=ex.split("Response")[0]
            renew_exc=[]
            if sentence_ not in excamples:
                j=j+1
                # print(ex)
                ex=ex.strip().split("Context:")
                # print(ex)

                for excample in ex:
                    if len(excample)!=0:
                     
                        excample=excample.split("Response:")
                        input_=excample[0]
                        response_=excample[1].strip()
                        hts=R_HT_mapping[response_]
                        ht=random.sample(hts,1)[0].split("|")

                        renew_input=ht[0] +""+input_+""+ ht[1]
                        renew_response=ht[0] +"|"+response_+"|"+ ht[1]
                        # print(renew_input)
                        # print(renew_response)
                        new_e="Context: "+renew_input+" Response: "+renew_response
                        renew_exc.append(new_e)
                       
        

                renew_exc=" ".join(renew_exc)
                new_instruction=instruction_+" "+renew_exc
                # print(new_instruction)
                line["instruction"]=new_instruction
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()

            else:
                line["instruction"]=line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()

print(j)