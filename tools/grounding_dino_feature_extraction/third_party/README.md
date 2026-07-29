# Third-party source

This directory contains the inference components required by the optional
Grounding DINO feature extractor.

- `GroundingDINO/` is derived from
  [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  (`v0.1.0-alpha2-62-g856dde2`) and is licensed under Apache-2.0. The model
  inference path is extended to return encoder memory and decoder query
  features, and the tokenizer loader supports the bundled offline BERT
  configuration. Modified source files carry a SegDINO3D modification notice.
- `segment_anything/` is derived from
  [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
  (`6fdee8f2727f4506cfbbe553e23b895e27956588`) and is licensed under
  Apache-2.0.
- `bert-base-uncased/` contains tokenizer and architecture configuration files
  from the Apache-2.0-licensed
  [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
  model repository. They are used for offline model construction; trained
  BERT parameters are restored from the Grounding DINO checkpoint.

The Grounding DINO and SAM license files are retained in their respective
source directories. Model checkpoints are not included.
