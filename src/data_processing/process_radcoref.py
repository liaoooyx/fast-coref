import argparse
import collections
import json
import os
import re

import conll
from transformers import AutoTokenizer, LlamaTokenizer, LongformerTokenizerFast
from utils import (
    BaseDocumentState,
    flatten,
    get_sentence_map,
    normalize_word,
    split_into_segments,
)


class OntoNotesDocumentState(BaseDocumentState):
    def __init__(self, key):
        super().__init__(key)
        self.clusters = collections.defaultdict(list)

    def final_processing(self):
        # populate clusters
        first_subtoken_index = -1
        for seg_idx, segment in enumerate(self.segment_info):
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else "-"

                # for OntoNotes they use "-", for RadCoref we use "_"
                if coref != "-" and coref != "_":
                    last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(first_subtoken_index)
                        else:
                            cluster_id = int(part[:-1])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append((start, last_subtoken_index))
        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        self.merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        self.sentence_map = get_sentence_map(self.segments, self.sentence_end)
        self.subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(self.subtoken_map), (num_words, len(self.subtoken_map))
        assert num_words == len(self.sentence_map), (num_words, len(self.sentence_map))

    def finalize(self):
        self.final_processing()
        num_words = len(flatten(self.segments))
        assert num_words == len(self.orig_subtoken_map), (
            num_words,
            len(self.orig_subtoken_map),
        )
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.merged_clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
            "orig_subtoken_map": self.orig_subtoken_map,
            "orig_tokens": self.tokens,
        }


def process_speaker(speaker):
    speaker = speaker.replace("_", " ")
    return (" ".join([token.capitalize() for token in speaker.split()])).strip()


def get_document(document_lines, args):
    document_state = OntoNotesDocumentState(document_lines[0])

    tokenizer = args.tokenizer
    word_idx = -1
    orig_word_idx = -1
    last_speaker = "-"
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12

            word_idx += 1
            orig_word_idx += 1
            word = normalize_word(row[3])
            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]

            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
                document_state.orig_subtoken_map.append(orig_word_idx)
        else:
            document_state.sentence_end[-1] = True

    split_into_segments(
        document_state,
        args.seg_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_partition(split, args):
    input_path = os.path.join(args.input_dir, "{}.conll".format(split))
    output_path = os.path.join(args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len))
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            document = get_document(document_lines, args)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args):
    tokenizer = args.tokenizer

    for split in ["dev", "test", "train"]:
        minimize_partition(split, args)


def get_tokenizer(model_str):
    if "longformer" in model_str:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_str, add_prefix_space=True)
    elif "llama" in model_str:
        tokenizer = LlamaTokenizer.from_pretrained(model_str, legacy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_str)

    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory.")
    parser.add_argument("--output_dir", type=str, required=True, default=None, help="Output directory.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument("--seg_len", default=4096, required=True, type=int, help="Max. segment length")

    args = parser.parse_args()

    assert os.path.exists(args.input_dir)

    print(f"Model: {args.model_name_or_path}, Segment length: {args.seg_len}")
    args.tokenizer = get_tokenizer(args.model_name_or_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


if __name__ == "__main__":
    minimize_split(parse_args())
