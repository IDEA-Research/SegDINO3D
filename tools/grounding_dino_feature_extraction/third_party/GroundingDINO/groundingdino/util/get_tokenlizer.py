"""Grounding DINO tokenizer loader.

Modified for SegDINO3D to support offline, config-only BERT construction.
"""

import os

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizerFast,
)


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(
                    type(text_encoder_type)
                )
            )
    print("final text_encoder_type: {}".format(text_encoder_type))
    return AutoTokenizer.from_pretrained(text_encoder_type)


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (
        os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)
    ):
        if os.path.isdir(text_encoder_type):
            model_files = (
                "pytorch_model.bin",
                "model.safetensors",
                "tf_model.h5",
                "flax_model.msgpack",
            )
            if not any(
                os.path.exists(os.path.join(text_encoder_type, name))
                for name in model_files
            ):
                # Grounding DINO's checkpoint contains the BERT parameters.
                return BertModel(BertConfig.from_pretrained(text_encoder_type))
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
