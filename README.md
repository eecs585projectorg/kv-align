# KV-Align

## Guide

### Environment Setup

Reference: We extend the [StreamingLLM Repository](https://github.com/mit-han-lab/streaming-llm) and adopt the same setup code.

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
```

### Generation
1. Scroll to the bottom of the file `kv_align/generation.py`.
2. Check that `key_model_path` and `value_model_path` point to the trained key and value networks. The directory `models` should have sample models you can use, or you can train your own (see [Training](#training)).
3. Paste the "Example Code for Generation" at the bottom of the same file. Edit the example code as appropriate and run the file.

### Latency
1. Follow first two instructions for [Generation](#generation).
2. Paste the "Example Code for Latency Evaluation" at the bottom of the same file. Edit the example code as appropriate and run the file.

### Perplexity
1. Scroll to the bottom of the file `kv_align/perplexity.py`.
2. Check that `key_model_path` and `value_model_path` point to the trained key and value networks.
3. Edit parameters as desired and run the code.

### Training
1. Run `kv_align/training_data.py` to obtain the training data.
2. Run `kv_align/get_model.py`.

## References

```
Xiao, Guangxuan, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. "Efficient streaming language models with attention sinks." arXiv preprint arXiv:2309.17453 (2023).
```
```
tomaarsen, Attention Sinks in Transformers for endless fluent generation, (2023), GitHub repository, https://github.com/tomaarsen/attention_sinks
```
```
tomaarsen. 2023. Bigcode architecture. attention_sinks. https://github.com/tomaarsen/attention_sinks/issues/21.
```
```
ChatGPT was utilized during the coding process, e.g., for model training code and plotting code.
```
