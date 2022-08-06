# Why we forked?
The repo was forked from [shtoshni/fast-coref](https://github.com/shtoshni/fast-coref) of version [38e96c1bb561773926cdcf244dc57b4586e46048](https://github.com/shtoshni/fast-coref/tree/38e96c1bb561773926cdcf244dc57b4586e46048).

1. We fine-tuned the fast-coref model on the [i2b2 2011 - Coreference](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) dataset.

2. We used this repo as the library to properly conver the i2b2 dataset format.

3. We didn't change the evaluation methods.

Please find the details of this repo from [here](https://github.com/shtoshni/fast-coref).

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
