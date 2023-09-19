from typing import Dict

import torch
import torch.nn as nn
import transformers
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2Model,
    LlamaModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    T5EncoderModel,
)

transformers.logging.set_verbosity_error()


class BaseDocEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(BaseDocEncoder, self).__init__()
        self.config = config

        gradient_checkpointing = False
        if config.finetune:
            gradient_checkpointing = True
            # if torch.cuda.is_available():
            #     memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (
            #         1024**3
            #     )
            #     if memory_in_gb > 40:
            #         # Enough memory to not require gradient checkpointing
            #         gradient_checkpointing = False

        model_str: str = config.transformer.model_str

        if "t5" in model_str.lower():
            self.lm_encoder: PreTrainedModel = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path=model_str,
                output_hidden_states=False,
            )
        elif "gpt2" in model_str.lower():
            self.lm_encoder: PreTrainedModel = GPT2Model.from_pretrained(
                pretrained_model_name_or_path=model_str,
                output_hidden_states=False,
            )
        elif "llama" in model_str.lower():
            self.lm_encoder: PreTrainedModel = LlamaModel.from_pretrained(
                pretrained_model_name_or_path=model_str,
                output_hidden_states=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            self.lm_encoder: PreTrainedModel = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_str,
                output_hidden_states=False,
                add_pooling_layer=False,
            )
        if gradient_checkpointing:
            self.lm_encoder.gradient_checkpointing_enable()

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_str, use_fast=True)
        if config.add_speaker_tokens:
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        config.speaker_start,
                        config.speaker_end,
                    ]
                }
            )

            self.lm_encoder.resize_token_embeddings(len(self.tokenizer))

        if not config.finetune:
            for param in self.lm_encoder.parameters():
                # Don't update encoder params
                param.requires_grad = False

        self.hidden_size: int = self.lm_encoder.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.lm_encoder.parameters()).device

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        return self.tokenizer

    def to_add_speaker_tokens(self) -> bool:
        return self.add_speaker_tokens

    def forward(self, document: Dict):
        raise NotImplementedError
