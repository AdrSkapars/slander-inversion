# !pip install huggingface_hub
# from huggingface_hub import notebook_login
# notebook_login()
# # <your Hugging Face token>


from datasets import load_dataset, load_from_disk

# dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")





# print("Length of full dataset:", len(dataset))

# num_data = 100
# def conditions(x):
#     # winner_vicuna = (x["winner"] == "model_a" and x["model_a"] == "vicuna-13b")
#     # winner_vicuna = winner_vicuna or (x["winner"] == "model_b" and x["model_b"] == "vicuna-13b")
#     # if not winner_vicuna:
#     #     return False

#     # if not x["winner"] != "tie (bothbad)":
#     #     return False

#     if not  x["language"] == "English":
#         return False

#     if not x["turn"] == 1:
#         return False

#     # no_harm1 = not x["toxic_chat_tag"]["roberta-large"]["flagged"]
#     # no_harm1 = no_harm1 and not x["toxic_chat_tag"]["t5-large"]["flagged"]
#     # if not no_harm1:
#     #     return False

#     # # no_harm2
#     # for is_harmful in x["openai_moderation"]["categories"].values():
#     #     if is_harmful:
#     #         return False

#     convo = "conversation_a" if x["winner"] == "model_a" else "conversation_b"
#     max_len = len(tokenizer(x[convo][0]['content']).input_ids) <= 64#15
#     max_len = max_len and len(tokenizer(x[convo][1]['content']).input_ids) <= 100
#     if not max_len:
#         return False

#     inputt = prompt_template(x[convo][0]['content'])
#     outputt = x[convo][1]['content']
#     if prob_of_sequence(inputt, outputt) == 0:
#         return False

#     return True

# # small_dataset = dataset.select([0, 10, 20, 30, 40, 50])
# # even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)

# processed_data = dataset.filter(lambda x: conditions(x))
# processed_data = processed_data.select_columns(["question_id", "winner", "conversation_a", "conversation_b"])
# processed_data = processed_data.shuffle(seed=42)
# print("Length after processing:", len(processed_data))
# # processed_data.save_to_disk("/content/drive/MyDrive/University/Year 4/LLM/Saves/processed_dataset2")



reloaded_data = load_from_disk("/content/drive/MyDrive/University/Year 4/LLM/Saves/processed_dataset")
processed_data = reloaded_data[:30]

# reloaded_data = load_from_disk("/content/drive/MyDrive/University/Year 4/LLM/Saves/processed_dataset2")
# processed_data = reloaded_data[:226]

print("Length of chosen subset:", len(processed_data["question_id"]))
user_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][0]['content']
assistant_text = lambda x: processed_data["conversation_a" if processed_data["winner"][x] == "model_a" else "conversation_b"][x][1]['content']
# print("\nExample:")
# print([user_text(0)])
# print([assistant_text(0)])