import json

with open("train_chuck_5_triplet_llama2_13b_right_0.json","r") as fr:
    for line in fr.readlines():
        instruction=json.loads(line.strip())["instruction"]
        print(instruction)
        print("--------")

        # break