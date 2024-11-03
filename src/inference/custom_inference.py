"""
Custom inference script based on model_inference.py
"""

import argparse
import json
import logging
import os
import sys

sys.path.append("/home/yuxiang/liao/workspace/fast-coref/src")

import torch
from inference.tokenize_doc import basic_tokenize_doc, tokenize_and_segment_doc
from model.entity_ranking_model import EntityRankingModel
from model.utils import action_sequences_to_clusters
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

PROJ_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.basename(__file__)
log_dir = os.path.join(PROJ_ROOT_DIR, "outputs", "logs")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(log_dir, f"{file_name}.log"), "w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

LOGGER = logging.getLogger("root")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(file_handler)
LOGGER.addHandler(console_handler)


@torch.no_grad()
def inference(model, input_dict, **kwargs):
    extra_output_dict = {}
    pred_mentions, _, _, pred_actions = model(input_dict, extra_output=extra_output_dict)
    idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

    subtoken_map = input_dict["subtoken_map"]
    orig_tokens = input_dict["orig_tokens"]
    clusters = []
    for idx_cluster in idx_clusters:
        cur_cluster = []
        for ment_start, ment_end in idx_cluster:
            tok_start, tok_end = subtoken_map[ment_start], subtoken_map[ment_end]
            tok_str = " ".join(orig_tokens[tok_start : tok_end + 1])
            cur_cluster.append((tok_start, tok_end, tok_str))
        clusters.append(cur_cluster)

    return {
        "input_dict": input_dict,
        "subtoken_idx_clusters": idx_clusters,
        "token_idx_clusters": [(subtoken_map[ment_start], subtoken_map[ment_end]) for idx_cluster in idx_clusters for ment_start, ment_end in idx_cluster],
        "token_clusters": clusters,
        "actions": pred_actions,
        "mentions": pred_mentions,
        **kwargs,
    }


def preprocess_tokens(tokenizer, max_segment_len, sent_tok_list):
    """The input is a list of sentences with a nested list of tokens: [[tok, ...], ...]"""
    return tokenize_and_segment_doc(sent_tok_list, tokenizer, max_segment_len=max_segment_len)


def preprocess_text(tokenizer, max_segment_len, text):
    """The input is a string"""
    import spacy

    basic_tokenizer = spacy.load("en_core_web_sm")
    basic_tokenized_doc = basic_tokenize_doc(text, basic_tokenizer)
    tokenized_doc = tokenize_and_segment_doc(
        basic_tokenized_doc,
        tokenizer,
        max_segment_len=max_segment_len,
    )
    return tokenized_doc


def load_model(model_path, encoder_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_path, "model.pth"), map_location=device)

    config = OmegaConf.create(checkpoint["config"])
    if encoder_path is not None:
        config.model.doc_encoder.transformer.model_str = encoder_path

    model = EntityRankingModel(config.model, config.trainer)
    model.load_state_dict(checkpoint["model"], strict=False)

    if config.model.doc_encoder.finetune:
        if encoder_name is None:
            doc_encoder_dir = os.path.join(model_path, config.paths.doc_encoder_dirname)
            model.mention_proposer.doc_encoder.lm_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=doc_encoder_dir)
            model.mention_proposer.doc_encoder.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=doc_encoder_dir)
        if torch.cuda.is_available():
            model.cuda()
    model.eval()

    return model, config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.from_bash:
        pass

    model_path = "/home/yuxiang/liao/resources/downloaded_models/coref_model_9b02_25_4/best"  # exp11
    encoder_path = "/home/yuxiang/liao/resources/downloaded_models/longformer_coreference_joint"

    model, config = load_model(model_path, encoder_path)
    max_segment_len = config.model.doc_encoder.transformer.max_segment_len
    tokenizer = model.mention_proposer.doc_encoder.tokenizer
    toks = [["A", "calcific", "density", "is", "seen", "projecting", "at", "the", "left", "lung", "base", "laterally", "."], ["The", "calcific", "density", "may", "reflect", "a", "granuloma", "."], ["The", "calcific", "density", "may", "reflect", "a", "sclerotic", "finding", "within", "the", "rib", "."], ["The", "calcific", "density", "may", "reflect", "an", "object", "external", "to", "the", "patient", "."]]

    input_dict = preprocess_tokens(tokenizer, max_segment_len, toks)
    out_dict = inference(model, input_dict, doc_key="123")
    print(out_dict)
    LOGGER.info(out_dict)
