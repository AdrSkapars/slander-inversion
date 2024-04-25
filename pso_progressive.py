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

######################## Embedding-based PSO
import operator
import random
import pickle
import numpy
import math
import torch.nn.functional as F
import torch
from deap import base, creator, tools
import gc
from datetime import datetime, timedelta
import time
# transformers.logging.set_verbosity_error()
# from deap import benchmarks


#>>>>>>>>>>>>>>>>>>>>>>> Algo Definitions
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
    smin=None, smax=None, best=None)
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def generate_static(size, smin, smax, p_static):
    part = creator.Particle(p_static)
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


#>>>>>>>>>>>>>>>>>>>>>>> Evaluation Function
def eval_inds(emb_list, in_batch_size, target_tokens, t5_batch_size, prompt_template=None, max_input_len=15, target_tok_id=0, ind_evals=None):
    # global autoencoder

    # Have to convert to list of strings to stay same with previous code
    emb_list_list = [list([e]) for e in emb_list]
    emb_tensor = torch.Tensor(emb_list_list).to("cuda")
    string_list = t5_decode(emb_tensor, max_out_length=max_input_len, t5_batch_size=t5_batch_size)
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

    for part, prob in zip(emb_list,string_probabilities):
        part.fitness.values = (prob,)


#>>>>>>>>>>>>>>>>>>>>>>> Initialisation Function
def initialise_embed_pop(start_population_as, sample_x, max_input_len, num_inputs):
    if start_population_as == "random":
        pop = toolbox.population(n=num_inputs)
    # elif start_population_as == "random_dataset":
    #     pop = []
    #     random_dataset = rand_data_model_generate(num_data=num_inputs, max_new_tokens=max_input_len)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(random_dataset[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "random_fluent":
    #     pop = []
    #     random_fluent = fluent_model_generate(num_generate=num_inputs, max_new_tokens=max_input_len)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(random_fluent[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "random_fluent_output":
    #     pop = []
    #     random_fluent_output = fluent_model_generate(seed=assistant_text(sample_x), num_generate=num_inputs, max_new_tokens=max_input_len)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(random_fluent_output[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "output":
    #     pop = []
    #     one_embedding = t5_encode(assistant_text(sample_x)).tolist()[0]
    #     for n in range(num_inputs):
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "output_paraphrase":
    #     pop = []
    #     output_paraphrase = paraphrase_model_generate(assistant_text(sample_x), num_paraphrase=num_inputs, max_new_tokens=max_input_len)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(output_paraphrase[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "output_synonym":
    #     pop = []
    #     output_synonym = synonyms_model_generate(assistant_text(sample_x), num_synonyms=num_inputs, max_new_tokens=max_input_len)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(output_synonym[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "inversion":
    #     pop = []
    #     inversion = inversion_model_generate(assistant_text(sample_x), num_inversions=1, max_new_tokens=max_input_len, sample=False)[0]
    #     one_embedding = t5_encode(inversion).tolist()[0]
    #     for n in range(num_inputs):
    #         pop.append(toolbox.particle_static(p_static=one_embedding))
    # elif start_population_as == "inversion_sample":
    #     pop = []
    #     inversion_sample = inversion_model_generate(assistant_text(sample_x), num_inversions=num_inputs, max_new_tokens=max_input_len, sample=True)
    #     for n in range(num_inputs):
    #         one_embedding = t5_encode(inversion_sample[n]).tolist()[0]
    #         pop.append(toolbox.particle_static(p_static=one_embedding))

    return pop



# # #>>>>>>>>>>>>>>>>>>>>>>> Main Loop
# # Run/ search params
# max_input_len = 64 #99 # Basically unbounded?
# num_inputs = 500#1000
# max_gens = 100#100#200 #999 # unbounded
# # max_minutes = 10
# stop_early = True

# # Initialisation params
# start_population_as = [
#     "random",
#     "random_dataset",
#     "random_fluent",
#     "random_fluent_output",
#     "output",
#     "output_paraphrase",
#     "output_synonym",
#     "inversion",
#     "inversion_sample"
# ][0]

# initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][0]
# initial_model_based_on = ["gens", "max_score"][1]
# initial_model_param = [[50, 100], [0.01, 0.1]][0][0]

# # Model params
# t5_batch_size = 200
# main_model_name = ["gpt2", "llama2"][0]
# if initial_model_name is None:
#     set_model_params(main_model_name)
#     using_main_model = True
# else:
#     set_model_params(initial_model_name)
#     using_main_model = False

# # Define algorithm operations
# toolbox = base.Toolbox()
# toolbox.register("particle_static", generate_static, size=embedding_size, smin=-0.5, smax=0.5) # For initialising as target output
# toolbox.register("particle", generate, size=embedding_size, pmin=-0.5, pmax=0.5, smin=-0.5, smax=0.5)
# toolbox.register("population", tools.initRepeat, list, toolbox.particle)
# toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

# # Set up recorded statistics
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("max", numpy.max)
# stats.register("min", numpy.min)
# stats.register("avg", numpy.mean)
# stats.register("std", numpy.std)


# def normal_pso(start_arg, end_arg):
#     global using_main_model

#     for x in range(start_arg, end_arg):
#         experiment_start_time = time.process_time()#datetime.now()

#         print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
#         print("Target output:", [assistant_text(x)])
#         print("\nCurrent target output:", [assistant_text(x)])
#         print("gen  nevals  time            max         min avg         std        ")


#         # Tokenize target sequence
#         target_seq = prefix + assistant_text(x)
#         target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
#         target_tokens = target_tokens[remove_prefix:]

#         # Update evaluate based on new target sequence
#         # toolbox.register("evaluate", benchmarks.h1)
#         toolbox.register("evaluate", eval_inds, t5_batch_size=t5_batch_size, in_batch_size=in_batch_size, target_tokens=target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)

#         # Initalise population
#         pop = initialise_embed_pop(start_population_as, x, max_input_len, num_inputs)

#         logbook = tools.Logbook()
#         logbook.header = ["gen", "nevals", "time"] + stats.fields
#         best = None

#         # best_so_far = 0
#         # best_so_far_text = None
#         for g in range(max_gens):
#             toolbox.evaluate(emb_list=pop)
#             for part in pop:
#                 # part.fitness.values = toolbox.evaluate(part)
#                 if not part.best or part.best.fitness < part.fitness:
#                     part.best = creator.Particle(part)
#                     part.best.fitness.values = part.fitness.values
#                 if not best or best.fitness < part.fitness:
#                     best = creator.Particle(part)
#                     best.fitness.values = part.fitness.values
#             for part in pop:
#                 toolbox.update(part, best)

#             # Gather all the fitnesses in one list and print the stats
#             logbook.record(gen=g+1, nevals=len(pop), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **stats.compile(pop))
#             print(logbook.__str__().split("\n")[-1])
#             # Print best ones
#             bestest = tools.selBest(pop, k=1000)
#             best_fitnesses = [ind.fitness.values[0] for ind in bestest]
#             best_list = [list([e]) for e in bestest]
#             best_tensor = torch.Tensor(best_list).to("cuda")
#             best_inds = t5_decode(torch.Tensor(best_tensor).to("cuda"), max_out_length=max_input_len, t5_batch_size=t5_batch_size)
#             print([[best_fitnesses[i], [best_inds[i]]] for i in range(len(bestest))])

#             # Check if should be stopping
#             if stop_early and best_fitnesses[0] == 1:
#                 break
#             # if timedelta(seconds=(time.process_time()-experiment_start_time)) >= timedelta(minutes=max_minutes):
#             #     break

#             # Check if should be switching from intial model to main model, if havent already
#             if not using_main_model:
#                 if initial_model_based_on == "gens":
#                     if (g+1) >= initial_model_param:
#                         set_model_params(main_model_name)
#                         using_main_model = True
#                 elif initial_model_based_on == "max_score":
#                     if best_fitnesses[0] >= initial_model_param:
#                         set_model_params(main_model_name)
#                         using_main_model = True

#>>>>>>>>>>>>>>>>>>>>>>> Main Loop proggressive
# Run/ search params
max_input_len = 64 #99 # Basically unbounded?
num_inputs = 500#20#1000
max_gens = 100 #200 #999 # unbounded
# max_minutes = 10
stop_early = True

# Initialisation params
start_population_as = [
    "random",
    "random_dataset",
    "random_fluent",
    "random_fluent_output",
    "output",
    "output_paraphrase",
    "output_synonym",
    "inversion",
    "inversion_sample"
][0]

initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][0]
initial_model_based_on = ["gens", "max_score"][1]
initial_model_param = [[50, 100], [0.01, 0.1]][0][0]

# Model params
t5_batch_size = 200
main_model_name = ["gpt2", "llama2"][0]
if initial_model_name is None:
    set_model_params(main_model_name)
    using_main_model = True
else:
    set_model_params(initial_model_name)
    using_main_model = False

# Proggressive params
progressive_based_on = ["gens", "non_zero_scores"][0]
progressive_param = 100

# Define algorithm operations
toolbox = base.Toolbox()
toolbox.register("particle_static", generate_static, size=embedding_size, smin=-0.5, smax=0.5) # For initialising as target output
toolbox.register("particle", generate, size=embedding_size, pmin=-1, pmax=1, smin=-0.5, smax=0.5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

# Set up recorded statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", numpy.max)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)


def proggressive_gens_pso(start_arg, end_arg):
    global using_main_model

    for x in range(start_arg, end_arg):
        experiment_start_time = time.process_time()#datetime.now()

        print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])

        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Initalise population
        pop = initialise_embed_pop(start_population_as, x, max_input_len, num_inputs)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "time"] + stats.fields
        best = None

        # Decide on progressive intervals over the total number of generations
        gen_slices = [progressive_param // len(target_tokens) for _ in range(len(target_tokens))]
        # Spread the leftover generations evenly starting from final slice and going backwards until run out (earlier slices likely won't get any extra)
        for slice_i in range(len(gen_slices)-1, (len(gen_slices)-1-(progressive_param % len(target_tokens))), -1):
            gen_slices[slice_i] += 1
        assert sum(gen_slices) == progressive_param

        gens_so_far = 0
        # best_so_far = 0
        # best_so_far_text = None
        for gen_i in range(len(gen_slices)):
            # gen_slice is the amount of generations to currently do
            gen_slice = gen_slices[gen_i]
            if gen_slice == 0:
                continue
            gens_so_far = sum(gen_slices[:gen_i])

            # Reveal another token of the target output for each gen_slice completed
            run_target_tokens = target_tokens[:gen_i+1]
            print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
            print("gen  nevals  time            max         min avg         std        ")

            if gens_so_far != 0:
                # All fitness functions need to be updated based on the new target output (if theyre non zero)
                # Smart method of having the probabilities be updated only by one extra target token
                invalid_ind = [ind for ind in pop if (not ind.fitness.valid) or (ind.fitness.values[0] != 0)]
                eval_inds(emb_list=invalid_ind, t5_batch_size=t5_batch_size, target_tok_id=gen_i, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)

                # Update the stats based on this partial update
                record = stats.compile(pop) if stats else {}
                logbook = tools.Logbook()
                logbook.header = ['gen', 'nevals', 'time'] + (stats.fields if stats else [])
                gen_addition = 0.5
                logbook.record(gen=(gens_so_far+gen_addition), nevals=len(invalid_ind), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **record)
                print(logbook.__str__().split("\n")[-1])
                # Print best ones
                bestest = tools.selBest(pop, k=1000)
                best_fitnesses = [ind.fitness.values[0] for ind in bestest]
                best_list = [list([e]) for e in bestest]
                best_tensor = torch.Tensor(best_list).to("cuda")
                best_inds = t5_decode(torch.Tensor(best_tensor).to("cuda"), max_out_length=max_input_len, t5_batch_size=t5_batch_size)
                print([[best_fitnesses[i], [best_inds[i]]] for i in range(len(bestest))])

            # Update evaluate based on new target sequence
            toolbox.register("evaluate", eval_inds, t5_batch_size=t5_batch_size, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)
            breaking = False
            for g in range(gen_slice):
                # Run PSO for one step
                toolbox.evaluate(emb_list=pop)
                for part in pop:
                    # part.fitness.values = toolbox.evaluate(part)
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if not best or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values
                for part in pop:
                    toolbox.update(part, best)

                # Gather all the fitnesses in one list and print the stats
                gens_so_far += 1
                logbook.record(gen=gens_so_far, nevals=len(pop), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **stats.compile(pop))

                # Print best ones
                print(logbook.__str__().split("\n")[-1])
                bestest = tools.selBest(pop, k=1000)
                best_fitnesses = [ind.fitness.values[0] for ind in bestest]
                best_list = [list([e]) for e in bestest]
                best_tensor = torch.Tensor(best_list).to("cuda")
                best_inds = t5_decode(torch.Tensor(best_tensor).to("cuda"), max_out_length=max_input_len, t5_batch_size=t5_batch_size)
                print([[best_fitnesses[i], [best_inds[i]]] for i in range(len(bestest))])

                # Check if should be stopping
                out_of_time = False #timedelta(seconds=(time.process_time()-experiment_start_time)) >= timedelta(minutes=max_minutes)
                out_of_time = out_of_time or (stop_early and best_fitnesses[0] == 1)
                if (gens_so_far >= max_gens) or out_of_time:
                    breaking = True
                    break

                # Check if should be switching from intial model to main model, if havent already
                if not using_main_model:
                    if initial_model_based_on == "gens":
                        if gens_so_far >= initial_model_param:
                            set_model_params(main_model_name)
                            using_main_model = True
                    elif initial_model_based_on == "max_score":
                        if best_fitnesses[0] >= initial_model_param:
                            set_model_params(main_model_name)
                            using_main_model = True

            if breaking:
                break

        # If still havent reached max gens/ time then continue searching with full target output
        for g in range(gens_so_far+1,max_gens+1):
            if out_of_time:
                break

            toolbox.evaluate(emb_list=pop)
            for part in pop:
                # part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            # gens_so_far += 1
            logbook.record(gen=g, nevals=len(pop), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **stats.compile(pop))

            # Print best ones
            print(logbook.__str__().split("\n")[-1])
            bestest = tools.selBest(pop, k=1000)
            best_fitnesses = [ind.fitness.values[0] for ind in bestest]
            best_list = [list([e]) for e in bestest]
            best_tensor = torch.Tensor(best_list).to("cuda")
            best_inds = t5_decode(torch.Tensor(best_tensor).to("cuda"), max_out_length=max_input_len, t5_batch_size=t5_batch_size)
            print([[best_fitnesses[i], [best_inds[i]]] for i in range(len(bestest))])

            # Check if should be stopping
            out_of_time = False #timedelta(seconds=(time.process_time()-experiment_start_time)) >= timedelta(minutes=max_minutes)
            out_of_time = out_of_time or (stop_early and best_fitnesses[0] == 1)

            # Check if should be switching from intial model to main model, if havent already
            if not using_main_model:
                if initial_model_based_on == "gens":
                    if gens_so_far >= initial_model_param:
                        set_model_params(main_model_name)
                        using_main_model = True
                elif initial_model_based_on == "max_score":
                    if best_fitnesses[0] >= initial_model_param:
                        set_model_params(main_model_name)
                        using_main_model = True



# normal_pso(0,30)
proggressive_gens_pso(0, 30)

# del autoencoder
del model
torch.cuda.empty_cache()
