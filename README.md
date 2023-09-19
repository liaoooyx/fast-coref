## Why we forked?
The repo is forked from [shtoshni/fast-coref](https://github.com/shtoshni/fast-coref) of version [38e96c1bb561773926cdcf244dc57b4586e46048](https://github.com/shtoshni/fast-coref/tree/38e96c1bb561773926cdcf244dc57b4586e46048).

1. We fine-tuned the fast-coref model on the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset.
2. We made minor changes to adapt to the MIMIC-CXR datasets.
3. The fine-tuned models can be downloaded [here](https://drive.google.com/drive/folders/1ZAVJYo9c5bobNeQdQexlOCoGzhF-u02G?usp=sharing), including EX12-best (F1=79.8), EX13-best (F1= 77.9), EX14-best (F1=79.5)
4. We used [shtoshni/longformer_coreference_joint](https://huggingface.co/shtoshni/longformer_coreference_joint) as the encoder.

## Requirements

We create our env with the following:

1. `conda create --name coref python=3.9 -y`
2. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
    1. `nvcc -V` to check cuda version
    2. `python` -> `import torch` -> `print(torch.__version__)` to check torch+cuda version
    3. Find the correct torch version+cuda from [here](https://pytorch.org/get-started/previous-versions/) to install (for us: pytorch1.12.1 + cuda11.3)
3. `pip install transformers` (4.33.1)
4. `pip install scipy` (1.11.2)
6. `pip install hydra-core --upgrade` (1.3.2)

### Optional 
1. `pip install sentencepiece` (0.1.99). Required for LLaMA2
2. `pip install accelerate` (0.23.0) Required to load a big model
3. `pip install peft` 
4. ``

## Preprocessing data

For GPT2
```shell
python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/lf \
--model_name_or_path /root/autodl-tmp/hg_offline_models/longformer_coreference_joint \
--seg_len 4096

python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/gpt2 \
--model_name_or_path /root/autodl-tmp/hg_offline_models/gpt2 \
--seg_len 1024

python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/flant5 \
--model_name_or_path /root/autodl-tmp/hg_offline_models/flan-t5-base \
--seg_len 512

python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/bert \
--model_name_or_path /root/autodl-tmp/hg_offline_models/bert-base-uncased \
--seg_len 512

python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/spanbert \
--model_name_or_path /root/autodl-tmp/hg_offline_models/spanbert-base-cased \
--seg_len 512

python /root/workspace/fast-coref/src/data_processing/process_radcoref.py \
--input_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/conll \
--output_dir /root/workspace/fast-coref/coref_resources/data/radcoref_475/llama2 \
--model_name_or_path /root/autodl-tmp/hg_offline_models/Llama-2-7b-hf \
--seg_len 4096
```

## Training

Check and modify config files in 
1. `src/conf/infra`
2. `src/conf/datasets`
3. `src/conf/model/doc_encoder/transformer`
4. `src/conf/experiment`
5. `src/conf/config.yaml`

```shell
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 /root/workspace/fast-coref/src/main.py \
infra=autodl \
experiment=autodl_comparison \
paths.model_name=exp6_flant5_2
```



## Issues

### Can't find model 'en_core_web_sm' 
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

```
pip install spacy
python -m spacy download en_core_web_sm
```
