# KV-Align

KV-Align is a new approach for generating an infinite output from a large language model, complementing the work of StreamingLLM [1, 2]. The main components of KV-Align are the key and value networks that realign positional information in the KV-cache upon token evictions, addressing the Github issue [5]. We provide a detailed guide for the codebase below, as well as a short list of coding [references](#references). A more detailed explanation of our approach and a thorough bibliography can be found in the paper submission for this project.

## Guide

You should have an NVIDIA GPU on your device to ensure the code does not error. All of the following instructions assume that you are running the command with `kv-align` as the current directory. This is important to ensure that certain files are saved and loaded to intended locations.

### Environment Setup

```bash
conda create -yn kv-align python=3.8
conda activate kv-align

pip install torch transformers==4.33.0 datasets nltk matplotlib
```

### Generation
The following command runs a generation of 2000 tokens using DistilGPT2 with cache size 54 using KV-Align. Again, make sure `kv-align` with the dash (not `kv_align` with the underscore) is your current directory.

```bash
python kv_align/generation.py --model_name distilgpt2 --num_tokens 2000 --cache_size 54 --flush --mode kv
```
Note that similarly as in [2], the generation is in response to the following default prompt from MT-Bench [4]: 
> Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.

You can manually change the prompt by modifying `kv_align/generation.py`.

Selected Parameters for Generation:
- `mode`: `kv`, `sw`, or `ar`, which correspond to KV-Align, StreamingLLM, and attention recomputation approaches, respectively. Other accepted synonyms for `mode` can be found in `kv_align/parse_model_mode.py`.
- `model_name`: valid inputs are `distilgpt2` and `smollm2`.  Other accepted synonyms for `model_name` can be found in `kv_align/parse_model_mode.py`.
- `allow_repeats`: If and only if this flag is used, `no_repeat_2gram = False`. 
- `flush`: If and only if this flag is used, we pass `flush=True` as an argument to `print`


### Latency Evaluation
```bash
python kv_align/generation.py --model_name gpt2 --latency --num_tokens 2000
```

### Perplexity Evaluation
```bash
python kv_align/perplexity.py --model_name gpt2 --dataset_name wikitext --num_tokens 20000 
```
Selected parameters for perplexity generation:
- `dataset_name`: valid inputs are `wikitext` and `everyday`


### Key and Value Network Training 
Note that the pretrained key and value networks are provided in `models` directory. If you would like to train your own network, you can follow the steps below:
1. Modify and run `kv_align/training_data.py` as appropriate (for your model of choice) to obtain the training data.
2. Modify and run `kv_align/key_value_network.py` to train the key and value networks. Alternatively, you can install Ray Tune and run the hyperparameter tuning with `kv_align/hyperparameter_tuning.py`.

### Code for Plots in the Paper
The codes ran to produce some of the final paper's plots can be found in the `plot` directory.

## References
We include a list of coding references at the end of this section. Here are some additional comments on the references:

- The official StreamingLLM paper and repository are [1] and [2], respectively, the latter of which was a key coding reference. Moreover, the file `modify_llama.py` originates from [2]. We also use a similar environment setup code as [2], and our perplexity and latency evaluations closely follow the approaches by the authors of [1] and [2].

- The example `prompts` in `generation.py` originate from MT-bench dataset of FastChat repository (see [4]). Note that [2] also uses prompts from [4].

- The difficulty of adapting GPT2 with StreamingLLM was observed in the Github issue [5]. The issue is from the Github repository [3] that also implements the StreamingLLM approach. The readability plot in this repository closely references the analogous plots in [3].

- With regards to [6], we note that ChatGPT was utilized astutely during the coding process, e.g., for model training code and plotting code.

```
[1] Xiao, Guangxuan, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. "Efficient streaming language models with attention sinks." ICLR, 2024.
```
```
[2] mit-han-lab. Efficient Streaming Language Models with Attention Sinks, (2024), GitHub repository, https://github.com/mit-han-lab/streaming-llm
```
```
[3] tomaarsen. Attention Sinks in Transformers for endless fluent generation, (2023), GitHub repository, https://github.com/tomaarsen/attention_sinks
```
```
[4] lm-sys. Fastchat, (2024), GitHub repository, https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```
```
[5] tomaarsen. 2023. Bigcode architecture. attention_sinks. https://github.com/tomaarsen/attention_sinks/issues/21
```
```
[6] OpenAI, 2024. ChatGPT (4o) [Large Language Model] https://chatgpt.com
```
