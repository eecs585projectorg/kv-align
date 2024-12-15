def parse_model_name(args):
    if args.model_name in ["distilbert/distilgpt2", "distilgpt2", "gpt2"]:
        model_name = "distilbert/distilgpt2"
    elif args.model_name in ["HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M", "SmolLM2", "smollm2", "rope"]:
        model_name = "HuggingFaceTB/SmolLM2-135M"
    else:
        raise ValueError("Invalid Model")
    return model_name
    
def parse_mode(args):
    if args.mode in ["key_value_net", "kvn", "kv_align", "kva", "kv"]:
        mode = "key_value_net"
    elif args.mode in ["sliding_window", "sw"]:
        mode = "sliding_window"  
    elif args.mode in ["with_recompute", "wr"]:
        mode = "with_recompute"
    else:
        raise ValueError("Invalid Mode")
    return mode