from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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