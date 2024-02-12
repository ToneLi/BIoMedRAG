import json
import re

# def get_ground():
#     all_=[]
#     with open("data/renew_test.json", "r", encoding="utf-8") as fr:
#         for line in fr.readlines():
#             line=line.strip()
#             line=json.loads(line)
#             all_.append([line["SUBJECT_TEXT"],line["PREDICATE"],line["OBJECT_TEXT"]])
#     return all_
        



# ground_gold=get_ground()


state_dict = {"p": 0, "c": 0, "g": 0}
def calclulate_f1(statics_dict, prefix=""):
    """
    Calculate the prec, recall and f1-score for the given state dict.
    The state dict contains predict_num, golden_num, correct_num.
    Reutrn a dict in the form as "prefx-recall": 0.99.
    """
    #{'p': 93, 'c': 34, 'g': 320}
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}


i=-1
with open("chuck_5_triplet_8000.json", "r", encoding="utf-8") as fr:  # llama2_7b_1000_KNN_0.json   llama2_7b_sample_1000.json
    for line in fr.readlines():
        line=line.strip()
        line=json.loads(line)
        i=i+1
        # print(line)
        P=set()
        sentence=line["sentence"].lower()
        gold_triples=set()
        ground_truth=line["ground_truth"]
       
    

        gold_t=ground_truth.split("|")
        gh=gold_t[0].lower().replace("       "," ")
        gr=gold_t[1].lower()
        gt=gold_t[2].lower().replace("       "," ")
      
        gold_triples.add((gh,gr,gt))
        # gold_triples.add((gh))
        # gold_triples.add((gt))
        # gold_triples.add((gr))
            
    

        predictions=line["predicted"].split("\n\n")[3].split("\n")


        if len(predictions)==2:
            
            predicted_=predictions[1]
            # print(predicted_)
            predicte_t=predicted_.split("|")
            if len(predicte_t)==3:
                ph=predicte_t[0].lower().replace("       "," ").strip()
           
                pt=predicte_t[2].lower().replace("       "," ").strip()#.replace(" .","").replace(" 3. context:{}. response: {}","").replace(" 1. context:{}. response: {}","").replace(" 2. context:{}. response: {}","").replace(" 3. response: {}","")
                pr=predicte_t[1].lower()
             
                P.add((ph,pr,pt))
                # P.add((ph))
                # P.add((pt))
                # P.add((pr))
           
  
      
        # if P!=gold_triples:
        #     print("pre",P)
        #     print("gold",gold_triples)
        #     print("-------------")
        # else:
        #      print("pre",P)
            
   
        state_dict["p"] += len(P)
        state_dict["g"] += len(gold_triples)
        state_dict["c"] += len(P & gold_triples)

#         # break
all_metirc_results = calclulate_f1(state_dict, 'all')

print(all_metirc_results)


"""
5000: {'all-prec': 0.7917570498915402, 'all-recall': 0.7849462365591398, 'all-f1': 0.7883369330453565}
8000: {'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}


triple:{'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}
h: {'all-prec': 0.928416485900217, 'all-recall': 0.9204301075268817, 'all-f1': 0.9244060475161987}
t:{'all-prec': 0.9175704989154013, 'all-recall': 0.9096774193548387, 'all-f1': 0.9136069114470842}
r:  {'all-prec': 0.8720173535791758, 'all-recall': 0.864516129032258, 'all-f1': 0.8682505399568036}
"""