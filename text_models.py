from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, logging
import torch
import torch.nn.functional as F

import random


model = None
tokenizer = None
eos_token = None
in_batch_size = None
prefix = None
remove_prefix = None
vocab_size = None
prompt_template = None

def load_model_with_params(model_name):
    global model, tokenizer, in_batch_size, prefix, remove_prefix, vocab_size, prompt_template

    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        in_batch_size = 400
        prefix = " "
        remove_prefix = 0
        vocab_size = 50257
        prompt_template = lambda x: f"User: {x}\nAssistant:"
    
    elif model_name == "gpt2-large":
        model = AutoModelForCausalLM.from_pretrained("gpt2-large").to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large", padding_side="left")
        in_batch_size = 200
        prefix = " "
        remove_prefix = 0
        vocab_size = 50257
        prompt_template = lambda x: f"User: {x}\nAssistant:"

    elif model_name == "llama2":
        model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ",
                    device_map="auto", trust_remote_code=False, revision="main")
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", use_fast=True)
        in_batch_size = 200
        prefix = "\n\n"
        remove_prefix = 4
        vocab_size = 32000
        prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"
    
    tokenizer.padding_side = "left"
    eos_token = tokenizer(tokenizer.bos_token).input_ids[0]
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.eos_token_id


def fluent_model_generate(seed=None, num_generate=1, batch_size=100, max_new_tokens=20):
    if seed is None:
        prefix_length = 2
    else:
        if seed[-1] == " ":
            seed = seed[:-1]
        if seed[-1] not in [".", "!", "?"]:
            seed += "."
        prefix_length = len(tokenizer(seed).input_ids)

    all_sequences = []
    for i in range(0, num_generate, batch_size):
        batch_s = min(batch_size, num_generate-i)
        if seed is None:
            prompt_batch = [tokenizer.decode(random.randint(0,vocab_size))+". " for _ in range(batch_s)]
        else:
            prompt_batch = [seed]*batch_s

        input_ids = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
        generate_output = model.generate(**input_ids, temperature=1, do_sample=True, top_p=0.95, top_k=100, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token)
        sequences = [seq[prefix_length:] for seq in generate_output.sequences]
        all_sequences += sequences

    random_tokens = lambda : tokenizer.decode([random.randint(0, vocab_size) for _ in range(random.randint(1, max_new_tokens))])
    # random_tokens = lambda : "None"

    decoded_sequences = tokenizer.batch_decode(all_sequences, skip_special_tokens=True)
    decoded_sequences = [random_tokens() if len(seq) == 0 else seq for seq in decoded_sequences]
    decoded_sequences = [seq[1:] if seq[0] in [" ", "\xa0", "\n"] else seq for seq in decoded_sequences]
    decoded_sequences = [seq[1:] if seq[0] in [" ", "\xa0", "\n"] else seq for seq in decoded_sequences]
    # decoded_sequences = [seq.replace("\xa0", "") for seq in decoded_sequences]

    return decoded_sequences


def prob_of_sequence(given_seq, test_seq):
    # Find probability of any sequence given context
    test_string = " " + test_seq
    test_tokens = tokenizer(test_string, return_tensors='pt').input_ids[0]
    # test_tokens = test_tokens[4:] # Remove '<s> \n\n'

    current_toks = []
    seq_prob = 1
    context = given_seq
    for t_tok in test_tokens:
        current_context = context + tokenizer.decode(current_toks)
        new_input_ids = tokenizer(current_context, return_tensors='pt').input_ids.cuda()
        new_generate_output = model.generate(inputs=new_input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=300, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token,)
        new_generate_output_scores = F.softmax(new_generate_output.scores[0][0], dim=0)
        t_score = new_generate_output_scores[t_tok]

        # # Just for test, get the distributions
        # distribution = []
        # nonzero_indices = torch.nonzero(new_generate_output_scores)
        # nonzero_values = new_generate_output_scores[nonzero_indices[:, 0]]
        # sorted_tensor = torch.sort(nonzero_values, descending=True)
        # sorted_indices = nonzero_indices[sorted_tensor.indices]
        # for i in range(len(sorted_indices)):
        #     distribution.append([tokenizer.decode(sorted_indices[i][0]), int(sorted_indices[i][0]), round(float(sorted_tensor.values[i]),4)])
        # print(tokenizer.decode(t_tok), "\t-", distribution)

        if t_score == 0:
            # print("\nProbability of sequence: 0 %")
            return 0

        seq_prob *= float(t_score)
        current_toks.append(t_tok)

    # print("Probability of sequence:", seq_prob * 100, "%")
    return seq_prob