## Why we forked?
The repo is forked from [shtoshni/fast-coref](https://github.com/shtoshni/fast-coref) of version [38e96c1bb561773926cdcf244dc57b4586e46048](https://github.com/shtoshni/fast-coref/tree/38e96c1bb561773926cdcf244dc57b4586e46048).

1. We fine-tuned the fast-coref model on the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset.
2. We made minor changes to adapt to the MIMIC-CXR datasets.
3. The fine-tuned models can be downloaded [here](https://drive.google.com/drive/folders/1ZAVJYo9c5bobNeQdQexlOCoGzhF-u02G?usp=sharing), including EX12-best (F1=79.8), EX13-best (F1= 77.9), EX14-best (F1=79.5)
4. We used [shtoshni/longformer_coreference_joint](https://huggingface.co/shtoshni/longformer_coreference_joint) as the encoder.

## Issues

### Can't find model 'en_core_web_sm' 
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

```
pip install spacy
python -m spacy download en_core_web_sm
```
