## PETAILOR: Improving Large Language Models with a Tailored Chunk Scorer for Biomedical Triple Extraction

### 1) Overview

The architecture of our proposed PeTailor is depicted in the diagram below.
It  consists of three major steps:  (1) constructing the diverse chunk database; (2) training the tailored chunk scorer to select the relevant document for the input sentence, the relevant document is from the diverse chunk database; (3) incorporating the retrieved document into the LLM  to generate the triple for the given sentence.
<img src="https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/blob/main/framework.png" width="800"/>

###  2) Baselines
 For the baseline models, please refer [Trple Extraction Baselines](https://github.com/ToneLi/Sunflowers-triplet-extraction)

### 3)[GM-CIHT](https://github.com/ToneLi/PETAILOR-for-bio-triple-extraction/tree/main/dataset/0_GM-CIHT)

data format:

```
{"PREDICATE": "INTERACTS_WITH",
"SUBJECT_TEXT": "nalorphine",
"OBJECT_TEXT": "morphine",
"SENTENCE": "[on the effect of respiration of a combination of  morphine -like acting pethidine with the  morphine  antagonist  nalorphine ]."}
```





