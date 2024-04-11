from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.corpus import wordnet
# nltk.download('wordnet')

import random
import json


### Inversion Model
inv_model = AutoModelForSeq2SeqLM.from_pretrained("wentingzhao/inversion_seq2seq_baseline").eval().to("cuda")
inv_tokenizer = AutoTokenizer.from_pretrained("wentingzhao/inversion_seq2seq_baseline")

inv_tokenizer.padding_side = "left"
inv_tokenizer.pad_token = inv_tokenizer.eos_token
inv_model.config.pad_token_id = inv_model.config.eos_token_id
inv_eos_token = inv_tokenizer(inv_tokenizer.eos_token).input_ids[0]

inv_model = None
torch.cuda.empty_cache()
del inv_model
torch.cuda.empty_cache()

def inversion_model_generate(target_output, num_inversions=1, batch_size=100, max_new_tokens=20, sample=False):
    # Load model
    inv_model = AutoModelForSeq2SeqLM.from_pretrained("wentingzhao/inversion_seq2seq_baseline").eval().to("cuda")

    outputs = []
    for i in range(0, num_inversions, batch_size):
        batch_s = min(batch_size, num_inversions-i)
        input_ids = inv_tokenizer([target_output]*batch_s, return_tensors="pt", padding=True).input_ids.to("cuda")

        if sample:
            output = inv_model.generate(
                inputs=input_ids,
                temperature=1,
                do_sample=True,
                top_p=0.99,
                top_k=500,
                max_new_tokens=max_new_tokens,
                pad_token_id=inv_eos_token,
            )
        else:
            output = inv_model.generate(
                inputs=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=inv_eos_token,
            )
        outputs += list(output)

    # Delete model
    inv_model = None
    torch.cuda.empty_cache()
    del inv_model
    torch.cuda.empty_cache()

    return [inv_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


### Paraphrase Model
paraph_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").eval().to('cuda')
paraph_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

paraph_model = None
torch.cuda.empty_cache()
del paraph_model
torch.cuda.empty_cache()

def paraphrase_model_generate(target_output, num_paraphrase=1, batch_size=100, max_new_tokens=20):
    # Load model
    paraph_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").eval().to('cuda')

    text =  "paraphrase: " + target_output + " </s>"
    encoding = paraph_tokenizer.encode_plus(text, padding='longest', return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = []
    for i in range(0, num_paraphrase, batch_size):
        batch_s = min(batch_size, num_paraphrase-i)

        output = paraph_model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            temperature=1.5,
            top_k=500,
            top_p=0.99,
            # early_stopping=True,
            max_length=max_new_tokens,
            num_return_sequences=batch_s
        )
        outputs += list(output)

    # Delete model
    paraph_model = None
    torch.cuda.empty_cache()
    del paraph_model
    torch.cuda.empty_cache()

    return [paraph_tokenizer.decode(outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in range(len(outputs))]


### Synonym Function
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name()
            synonyms.add(synonym)
    return list(synonyms)

def synonyms_model_generate(target_output, num_synonyms=1, max_new_tokens=20):
    words = target_output.split()
    variations = []
    for _ in range(num_synonyms):
        new_sentence = []
        for word in words:
            synonyms = get_synonyms(word)
            if synonyms:
                new_sentence.append(random.choice(synonyms).replace("_", " "))
            else:
                new_sentence.append(word)
        variations.append(' '.join(new_sentence[:max_new_tokens]))
    return variations


### Random Data Function
# with open("dataset_tweet-scraper_2024-03-04_15-49-10-416.json", encoding="utf-8") as f:
#     ood_data = json.load(f)
def rand_data_model_generate(num_data=1, max_new_tokens=20):
    ood_sample = random.sample(ood_data, num_data)
    ood_sample = [tokenizer.decode(tokenizer(sample["text"]).input_ids[:max_new_tokens]) for sample in ood_sample]
    return ood_sample
