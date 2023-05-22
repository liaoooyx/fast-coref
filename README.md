## Why we forked?
The repo is forked from [shtoshni/fast-coref](https://github.com/shtoshni/fast-coref) of version [38e96c1bb561773926cdcf244dc57b4586e46048](https://github.com/shtoshni/fast-coref/tree/38e96c1bb561773926cdcf244dc57b4586e46048).

1. We fine-tuned the fast-coref model on the [i2b2 2011 - Coreference](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) dataset and the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset.
2. We made minor changes to adapt to the i2b2 and MIMIC-CXR datasets.
3. We didn't change the evaluation methods.

Please find the details of this repo from [here](https://github.com/shtoshni/fast-coref).

## Issues

### Can't find model 'en_core_web_sm' 
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

```
pip install spacy
python -m spacy download en_core_web_sm
```

### Citation
```
@inproceedings{toshniwal2021generalization,
    title = {{On Generalization in Coreference Resolution}},
    author = "Shubham Toshniwal and Patrick Xia and Sam Wiseman and Karen Livescu and Kevin Gimpel",
    booktitle = "CRAC (EMNLP)",
    year = "2021",
}

@inproceedings{toshniwal2020bounded,
    title = {{Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks}},
    author = "Shubham Toshniwal and Sam Wiseman and Allyson Ettinger and Karen Livescu and Kevin Gimpel",
    booktitle = "EMNLP",
    year = "2020",
}
```