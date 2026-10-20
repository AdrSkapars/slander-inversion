import gc

########################################################################
######################## LOAD GENERATIVE MODEL
# GPT 2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()

gpt2_tokenizer.padding_side = "left"
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
eos_token = gpt2_tokenizer(gpt2_tokenizer.eos_token).input_ids[0]
# gpt2_tokenizer.pad_token = gpt2_tokenizer.bos_token
gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id


model = None
tokenizer = None
in_batch_size = None
prefix = None
remove_prefix = None
vocab_size = None
prompt_template = None

def set_model_params(model_name):
    global model, tokenizer, in_batch_size, prefix, remove_prefix, vocab_size, prompt_template

    if model_name == "gpt2":
        model = gpt2_model
        tokenizer = gpt2_tokenizer
        in_batch_size = 350
        prefix = " "
        remove_prefix = 0
        vocab_size = 50257
        prompt_template = lambda x: f"User: {x}\nAssistant:"

    elif model_name == "llama2":
        model = llama2_model
        tokenizer = llama2_tokenizer
        in_batch_size = 200
        prefix = "\n\n"
        remove_prefix = 4
        vocab_size = 32000
        prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"


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


# Find probability of any sequence given context
def prob_of_sequence(context, test_seq):
    prompt_template = lambda x: f"User: {x}\nAssistant:"

    test_string = " " + test_seq
    test_tokens = tokenizer(test_string, return_tensors='pt').input_ids[0]
    # test_tokens = test_tokens[4:] # Remove '<s> \n\n'

    current_toks = []
    seq_prob = 1
    for t_tok in test_tokens:
        current_context = context + tokenizer.decode(current_toks)
        new_input_ids = tokenizer(current_context, return_tensors='pt').input_ids.cuda()
        new_generate_output = model.generate(inputs=new_input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=300, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token,)
        new_generate_output_scores = F.softmax(new_generate_output.scores[0][0], dim=0)
        t_score = new_generate_output_scores[t_tok]

        # Just for test, distributions
#         distribution = []
#         nonzero_indices = torch.nonzero(new_generate_output_scores)
#         nonzero_values = new_generate_output_scores[nonzero_indices[:, 0]]
#         sorted_tensor = torch.sort(nonzero_values, descending=True)
#         sorted_indices = nonzero_indices[sorted_tensor.indices]
#         for i in range(len(sorted_indices)):
#             distribution.append([tokenizer.decode(sorted_indices[i][0]), int(sorted_indices[i][0]), round(float(sorted_tensor.values[i]),4)])
#         print(tokenizer.decode(t_tok), "\t-", distribution)

        if t_score == 0:
            # print("\nProbability of sequence: 0 %")
            return 0

        seq_prob *= float(t_score)
        current_toks.append(t_tok)

    # print("Probability of sequence:", seq_prob * 100, "%")
    return seq_prob


########################################################################
######################## LOAD EMBEDDING MODEL
class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)#, model_max_length=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
autoencoder = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-small-wikipedia', device=device)
embedding_size = 512

t5_dummy_tokens = autoencoder.tokenizer(".", return_tensors='pt').to(device)['input_ids']

def t5_encode(text, t5_decoder_inputs=t5_dummy_tokens):
    with torch.no_grad():
        inputs = autoencoder.tokenizer(text, return_tensors='pt').to(device)#, padding=True, truncation=True).to(device)
        outputs = autoencoder.model(**inputs, decoder_input_ids=t5_decoder_inputs, encode_only=True)
        # torch.cuda.empty_cache()
        del inputs # cleaning
        return outputs

def t5_encode_list(texts):
    embeddings = []
    for text in texts:
        emb = t5_encode(text)#[0]
        embeddings.append(emb)
    return torch.stack(embeddings, dim=0)

t5_dummy_embed = t5_encode(".")

def t5_decode(embeddings, t5_batch_size, max_out_length=15, t5_dummy_tokens=t5_dummy_tokens, t5_dummy_embed=t5_dummy_embed, sample=False):
    with torch.no_grad():
        decoded_output = []
        for batch in range(0,len(embeddings),t5_batch_size):
            # if t5_batch_size != 1:
                # print(batch+t5_batch_size, ",", end="")
            current_embeddings = embeddings[batch:batch+t5_batch_size]
            autoencoder.model.perturb_vector = current_embeddings - t5_dummy_embed.unsqueeze(0)
            input_ids = t5_dummy_tokens.repeat(current_embeddings.shape[0], 1)  # Repeat the dummy text for each latent
            if not sample:
                outputs = autoencoder.model.generate(
                    input_ids=input_ids,
                    max_length=max_out_length,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    num_return_sequences=1
                )
            else:
                outputs = autoencoder.model.generate(
                    input_ids=input_ids,
                    max_length=max_out_length,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=1
                )
            decoded_output += [autoencoder.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            del current_embeddings # cleaning
            del input_ids # cleaning
            del outputs # cleaning
        # torch.cuda.empty_cache()
        return decoded_output


########################################################################
######################## LOAD DATASETS
# Chatbot Arena
from datasets import load_dataset, load_from_disk

reloaded_data = load_from_disk("processed_dataset")
processed_data = reloaded_data[:30]
user_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][0]['content']
assistant_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][1]['content']


########################################################################
######################## DEFINE ALGORITHMS

######################## Embedding-Based Metropolis-Hastings Sampling
import random
import math
import numpy
import copy
from numpy.random import choice

import gc


#>>>>>>>>>>>>>>>>>>>>>>> Evaluation Function
def eval_inds(emb_list, in_batch_size, target_tokens, t5_batch_size, prompt_template=None, target_tok_id=0, ind_evals=None):
    # global autoencoder

    # Have to convert to list of strings to stay same with previous code
    emb_list_list = [list([e]) for e in emb_list]
    emb_tensor = torch.Tensor(emb_list_list).to("cuda")
    # print("starting")
    string_list = t5_decode(emb_tensor, max_out_length=max_input_len, t5_batch_size=t5_batch_size)
    # print("ended")
    # del emb_tensor
    # # del autoencoder
    # gc.collect()
    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.empty_cache()
    # # autoencoder = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-small-wikipedia', device=device)
    # print("cleaned")

    if prompt_template is not None:
        string_list = [prompt_template(s) for s in string_list]

    num_strings = len(string_list)
    batch_size = min(in_batch_size, num_strings)
    # Initialise batch
    batch_string_ids = list(range(0,batch_size))
    next_string_to_batch = batch_size

    # Initialise string trackers
    if target_tok_id == 0 or ind_evals == None:
        string_probabilities = [1]*num_strings
        string_target_tok_id = [0]*num_strings
    else:
        # Individuals have already gotten probabilities for some tokens
        string_probabilities = [eval for eval in ind_evals]
        string_target_tok_id = [target_tok_id for _ in ind_evals]

    successful_string_ids = []

    # Go until there are no more strings being checked nor yet to be checked
    empty_batch_slots = 0
    counter = 0
    while empty_batch_slots != batch_size:
        # Get batch ready for inference
        batch_strings = []
        for b_id in range(batch_size):
            if b_id >= num_strings:
                break
            s_id = batch_string_ids[b_id]
            if s_id == -1:
                continue
            # Need to add target tokens checked so far to end of string
            b_string = string_list[batch_string_ids[b_id]]
            b_string += tokenizer.decode(target_tokens[:string_target_tok_id[s_id]])
            batch_strings.append(b_string)

        # Do the inference
        input_ids = tokenizer(batch_strings, return_tensors="pt", padding=True).to("cuda")
        generate_output = model.generate(**input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=300, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=eos_token,)
        # generate_output = model.generate(**input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)

        # Update validity of strings based on inference
        empty = 0
        for b_id in range(batch_size):
            s_id = batch_string_ids[b_id]
            if s_id == -1:
                empty += 1
                continue
            curr_target_tok = target_tokens[string_target_tok_id[s_id]]
            prob = generate_output.scores[0][b_id-empty][curr_target_tok]

            if not (math.isinf(prob) and prob < 0):
                # String still viable candidate
                # Update probability
                True_distribution = F.softmax(generate_output.scores[0][b_id-empty], dim=0)
                string_probabilities[s_id] *= float(True_distribution[curr_target_tok])

                # Prepare to check for next target token
                string_target_tok_id[s_id] += 1
                if string_target_tok_id[s_id] == len(target_tokens):
                    # Have already checked all the target tokens
                    successful_string_ids.append(s_id)
                    # Finished so pretend string is not viable so it gets replaced
                    prob = -math.inf
            else:
                # String not a viable candidate
                # Update probability
                string_probabilities[s_id] = 0

            if math.isinf(prob) and prob < 0:
                # Replace non-viable and finished strings in batch
                if next_string_to_batch >= num_strings:
                    # No new strings to start checking
                    batch_string_ids[b_id] = -1
                    empty_batch_slots += 1
                else:
                    batch_string_ids[b_id] = next_string_to_batch
                    next_string_to_batch += 1
        counter += 1

    return string_probabilities


#>>>>>>>>>>>>>>>>>>>>>>> Mutate Function
def mutate_sample(individual, mutation_size):
    # Propose on a copy so that a rejected proposal leaves the current state unchanged
    individual = list(individual)

    # Change each dimension of embedding
    for d in range(len(individual)):
        # Add small amount of gausian noise while staying between -1 and 1
        mut_sum = individual[d] + numpy.random.normal(0, mutation_size)
        individual[d] = min(max(mut_sum, -1), 1)
    return individual


#>>>>>>>>>>>>>>>>>>>>>>> Main Loop
# Parameters which are manually defined
max_input_len = 64 #99 # Basically unbounded?
num_inputs = 1000
total_gens = 200
mutation_size = 0.1
# embedding_size

# Parameters which are model specific
t5_batch_size = 200
model_name = "gpt2"
set_model_params(model_name)

if model_name == "gpt2":
    in_batch_size = 200
    prefix = " "
    remove_prefix = 0
    vocab_size = 50257
    prompt_template = lambda x: f"User: {x}\nAssistant:"
if model_name == "llama2":
    in_batch_size = 200
    prefix = "\n\n"
    remove_prefix = 4
    vocab_size = 32000
    prompt_template = lambda x: f"[INST]{x}[/INST]\n\n"

def normal_emb_sample(start_arg, end_arg):
    for x in range(start_arg, end_arg):
        print("-"*30)
        print("\n\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Initialise 'population'
        pop = [[random.uniform(-1, 1) for _ in range(embedding_size)] for _ in range(num_inputs)]
        pop_eval = eval_inds(pop, in_batch_size, target_tokens, t5_batch_size, prompt_template, 0)

        best_max = max(pop_eval)
        best_input = t5_decode(torch.unsqueeze(torch.Tensor(pop[numpy.argmax(pop_eval)]), dim=0).to("cuda"), max_out_length=max_input_len, t5_batch_size=1)

        for gen in range(total_gens):
            # Get new candidates (based on a single mutation)
            pop2 = [mutate_sample(individual, mutation_size) for individual in pop]
            # Work out acceptence based on evaluations
            pop_eval2 = eval_inds(pop2, in_batch_size, target_tokens, t5_batch_size, prompt_template, 0)
            # Change pop based on whether accepted
            for ind_i in range(len(pop)):
                acceptence = min(1,pop_eval2[ind_i]/pop_eval[ind_i]) if pop_eval[ind_i] != 0 else 1
                if acceptence != 0 and (acceptence == 1 or random.random() < acceptence):
                    pop[ind_i] = pop2[ind_i]
                    pop_eval[ind_i] = pop_eval2[ind_i]

            # Print best inputs of this gen and so far
            curr_max = max(pop_eval)
            curr_input = t5_decode(torch.unsqueeze(torch.Tensor(pop[numpy.argmax(pop_eval)]), dim=0).to("cuda"), max_out_length=max_input_len, t5_batch_size=1)
            print(gen, "\t", curr_max, "\t", curr_input)
            if curr_max > best_max:
                best_max = curr_max
                best_input = curr_input
            if best_max != 0:
                print("max", "\t", best_max, "\t", best_input)


normal_emb_sample(0, 30)
