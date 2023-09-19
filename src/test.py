import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

if __name__ == "__main__":
    # Load base model
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path="/root/autodl-tmp/hg_offline_models/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
    )
    print(model)
    pass
