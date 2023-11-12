from diffusers import DiffusionPipeline,DDPMScheduler
import pandas as pd
import os
import glob
import torch
import random


def load_XLembedding(base,token="my",embedding_file="myToken.pt",path="./Embeddings/"):
    emb=torch.load(path+embedding_file)
    set_XLembedding(base,emb,token)
    
def set_XLembedding(base,emb,token="my"):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=base.components["tokenizer_2"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
        tokenNo2=tokens[1]
        embs=base.components["text_encoder"].text_model.embeddings.token_embedding.weight
        embs2=base.components["text_encoder_2"].text_model.embeddings.token_embedding.weight
        assert embs[tokenNo].shape==emb["emb"].shape, "different 'text_encoder'"
        assert embs2[tokenNo2].shape==emb["emb2"].shape, "different 'text_encoder_2'"
        embs[tokenNo]=emb["emb"].to(embs.dtype).to(embs.device)
        embs2[tokenNo2]=emb["emb2"].to(embs2.dtype).to(embs2.device)
        
        
base_model_path="stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, #torch.bfloat16
    variant="fp32", 
    use_safetensors=True,
    add_watermarker=False,
)
pipe.enable_xformers_memory_efficient_attention()
torch.set_grad_enabled(False)
_=pipe.to("cuda:1")


df = pd.read_csv('prompts/prompts-personreal.csv')
prompts = list(df.prompt)
seeds = list(df.evaluation_seed)
case_numbers = list(df.case_number)

learned="sks"
embs_path="./textualinversion_models/"
emb_file="eyesize_textual_inversion.pt"

load_XLembedding(pipe,token=learned,embedding_file=emb_file,path=embs_path)

p1="photo of a person, realistic, 8k with {} eyes"
n_steps=50

seed = random.randint(0,2**15)
sample_prompt = p1
prompt=sample_prompt.format(learned)


for idx, prompt in enumerate(prompts):
    case_number = case_numbers[idx]
    seed = seeds[idx]
    
    print(prompt, seed)
    with torch.no_grad():    
        generator = torch.manual_seed(seed)
        images = pipe(
            prompt=prompt+ ' with sks eyes',
            num_inference_steps=n_steps,
            num_images_per_prompt=5,
            generator = generator
        ).images
    for i, im in enumerate(images):
        im.save(f'/share/u/rohit/www/textualinversion/eyesize_xl/{case_number}_{i}.png')