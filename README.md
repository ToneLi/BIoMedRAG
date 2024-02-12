## PETAILOR: Improving Large Language Models with a Tailored Chunk Scorer for Biomedical Triple Extraction

### 1) Overview

The architecture of our proposed PeTailor is depicted in the diagram below.
It  consists of three major steps:  (1) constructing the diverse chunk database; (2) training the tailored chunk scorer to select the relevant document for the input sentence, the relevant document is from the diverse chunk database; (3) incorporating the retrieved document into the LLM  to generate the triple for the given sentence.
<img src="https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/blob/main/framework.png" width="800"/>

###  2) Baselines
 For the baseline models, please refer [Trple Extraction Baselines](https://github.com/ToneLi/Sunflowers-triplet-extraction)

### 3) [GM-CIHT](https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/tree/main/dataset/0_GM-CIHT)

data format:

```
{"PREDICATE": "INTERACTS_WITH",
"SUBJECT_TEXT": "nalorphine",
"OBJECT_TEXT": "morphine",
"SENTENCE": "[on the effect of respiration of a combination of  morphine -like acting pethidine with the  morphine  antagonist  nalorphine ]."}
```
###  4) Code Structure

```
----0_make_relation_chuck_and_scorer_data (data preprogress)

----1_train_scorer_model  (chuck score training)
      --  please replace the trainer.py file in the source transformer file, and add the  SplitInputsChunks.py and ChuckWeights.py to the transformer file.

----2_relation_data_to_triple_train_data

----3_trainning_triple_model  (LLM training for triple extraction)

----4_generation_triple_model  (generation progress)
```

### 5) Configuration

1) Python  3.8.8

2) Transformer: pip install transformers==4.31.0

3) GPU A100

### 3)  Easy way to train the model and doing the model evalation

step1: enter the "https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/tree/main/3_trainning_triple_model" and run:

 ```
CUDA_VISIBLE_DEVICES=0  nohup  python trainer.py >myout.trainer 2>&1 &   
```

step 2:  The generated file is in [petailor_output_for_GM-CIHT](https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/blob/main/4_generation_triple_model/chuck_5_triplet_8000.json), please run:

 ```
  python   0_F1_triplet_evalution.py
  results: {'all-prec': 0.8177874186550976, 'all-recall': 0.810752688172043, 'all-f1': 0.8142548596112312}
```
 

### 4) How to run (full step)


```
Step 1: Please access the "0_make_relation_chuck_and_scorer_data" directory and execute the code, proceeding through the files sequentially according to their assigned numbers. Ensure to update the file names and locations as necessary.

Step2:  Please access the "1_train_scorer_model" directory and execute the code
       CUDA_VISIBLE_DEVICES=1 python 0_train_retrievel_5..py,  please use the default parameters.

Step3:   Please access the "2_relation_data_to_triple_train_data" directory and execute the code
       python chuck_triplet_progress_train.py

Step4:  Please access the "3_trainning_triple_model" directory and execute the code:
     CUDA_VISIBLE_DEVICES=1 python 1_ourmethod_chuck5_sim_llama2_13b_right_2.py

Step5:  Please access the "4_generation_triple_model" directory and execute the code:
       CUDA_VISIBLE_DEVICES=1 python  chuck5_generation.py

Step6:   Please access the "4_generation_triple_model" directory and execute the code  for the evaluation:
         python 0_F1_triplet_evalution.py
```

