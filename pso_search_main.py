# Run/ search params
max_input_len = 64 #99 # Basically unbounded?
num_inputs = 500#1000
max_gens = 50#100#200 #999 # unbounded
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
][4]

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

# Define algorithm operations
toolbox = base.Toolbox()
toolbox.register("particle_static", generate_static, size=embedding_size, smin=-0.5, smax=0.5) # For initialising as target output
toolbox.register("particle", generate, size=embedding_size, pmin=-0.5, pmax=0.5, smin=-0.5, smax=0.5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

# Set up recorded statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", numpy.max)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)


def normal_pso(start_arg, end_arg):
    global using_main_model

    for x in range(start_arg, end_arg):
        experiment_start_time = time.process_time()#datetime.now()

        print("\n\n"+"-"*50+"\n"+"-"*50+"\nOriginal input:", [user_text(x)])
        print("Target output:", [assistant_text(x)])
        print("\nCurrent target output:", [assistant_text(x)])
        print("gen	nevals	time          	max        	min	avg        	std        ")


        # Tokenize target sequence
        target_seq = prefix + assistant_text(x)
        target_tokens = tokenizer(target_seq, return_tensors='pt').input_ids[0]
        target_tokens = target_tokens[remove_prefix:]

        # Update evaluate based on new target sequence
        # toolbox.register("evaluate", benchmarks.h1)
        toolbox.register("evaluate", eval_inds, t5_batch_size=t5_batch_size, in_batch_size=in_batch_size, target_tokens=target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)

        # Initalise population
        pop = initialise_embed_pop(start_population_as, x, max_input_len, num_inputs)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "time"] + stats.fields
        best = None

        # best_so_far = 0
        # best_so_far_text = None
        for g in range(max_gens):
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
            logbook.record(gen=g+1, nevals=len(pop), time=timedelta(seconds=(time.process_time()-experiment_start_time)), **stats.compile(pop))
            print(logbook.__str__().split("\n")[-1])
            # Print best ones
            bestest = tools.selBest(pop, k=1000)
            best_fitnesses = [ind.fitness.values[0] for ind in bestest]
            best_list = [list([e]) for e in bestest]
            best_tensor = torch.Tensor(best_list).to("cuda")
            best_inds = t5_decode(torch.Tensor(best_tensor).to("cuda"), max_out_length=max_input_len, t5_batch_size=t5_batch_size)
            print([[best_fitnesses[i], [best_inds[i]]] for i in range(len(bestest))])

            # Check if should be stopping
            if stop_early and best_fitnesses[0] == 1:
                break
            # if timedelta(seconds=(time.process_time()-experiment_start_time)) >= timedelta(minutes=max_minutes):
            #     break

            # Check if should be switching from intial model to main model, if havent already
            if not using_main_model:
                if initial_model_based_on == "gens":
                    if (g+1) >= initial_model_param:
                        set_model_params(main_model_name)
                        using_main_model = True
                elif initial_model_based_on == "max_score":
                    if best_fitnesses[0] >= initial_model_param:
                        set_model_params(main_model_name)
                        using_main_model = True



# # Run/ search params
# max_input_len = 15#64 #99 # Basically unbounded?
# num_inputs = 50#20#1000
# max_gens = 5 #200 #999 # unbounded
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
# ][2]

# initial_model_name = [None, "gpt2", "gpt2-large", "llama2"][1]
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

# # Proggressive params
# progressive_based_on = ["gens", "non_zero_scores"][1]
# progressive_param = 1

# # Define algorithm operations
# toolbox = base.Toolbox()
# toolbox.register("particle_static", generate_static, size=embedding_size, smin=-0.5, smax=0.5) # For initialising as target output
# toolbox.register("particle", generate, size=embedding_size, pmin=-1, pmax=1, smin=-0.5, smax=0.5)
# toolbox.register("population", tools.initRepeat, list, toolbox.particle)
# toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

# # Set up recorded statistics
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("max", numpy.max)
# stats.register("min", numpy.min)
# stats.register("avg", numpy.mean)
# stats.register("std", numpy.std)


def proggressive_nzs_pso(start_arg, end_arg):
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

        # Initialise current target output
        target_tok_id = 0
        target_updated = False
        run_target_tokens = target_tokens[:target_tok_id+1]
        print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
        print("gen	nevals	time          	max        	min	avg        	std        ")
        toolbox.register("evaluate", eval_inds, t5_batch_size=t5_batch_size, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)

        # Run search until out of resources
        gens_so_far = 0
        out_of_time = False
        while not ((gens_so_far >= max_gens) or out_of_time):
            if target_updated:
                # Update current target output
                run_target_tokens = target_tokens[:target_tok_id+1]
                print("\nCurrent target output:", [tokenizer.decode(run_target_tokens)])
                print("gen	nevals	time          	max        	min	avg        	std        ")
                toolbox.register("evaluate", eval_inds, t5_batch_size=t5_batch_size, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template)

                # All fitness functions need to be updated based on the new target output (if theyre non zero)
                # Smart method of having the probabilities be updated only by one extra target token
                invalid_ind = [ind for ind in pop if (not ind.fitness.valid) or (ind.fitness.values[0] != 0)]
                eval_inds(emb_list=invalid_ind, t5_batch_size=t5_batch_size, target_tok_id=target_tok_id, in_batch_size=in_batch_size, target_tokens=run_target_tokens, prompt_template=prompt_template, max_input_len=max_input_len)

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

                target_updated = False

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

            # Check if should start considering more tokens, if percentage of non zero values in population are high enough
            if (gens_so_far < max_gens) and not out_of_time and (len(run_target_tokens) != len(target_tokens)):
                non_zeros = [1 for ind in pop if ind.fitness.values[0] != 0]
                non_zeros_percent = ((sum(non_zeros)/num_inputs)*100)
                if non_zeros_percent >= progressive_param:
                    target_tok_id += len(target_tokens) // 10 # Start considering next 10% of tokens
                    target_updated = True


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
            print("gen	nevals	time          	max        	min	avg        	std        ")

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
