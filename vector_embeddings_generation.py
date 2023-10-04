# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import csv
import json
import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open('/content/drive/MyDrive/Colab Notebooks/llama-main/myData.json') as f:
        news = [json.load(f)[2]]


    with open('/content/drive/MyDrive/Colab Notebooks/vect_embeddings.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, ['text', 'vect_embedding'])
        dict_writer.writeheader()


    prompts_size = len(news)-1
    for iter_num, new in enumerate(news):
        prompt = [new['content']]
        result = generator.text_completion(
            prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print(result)
        with open('/content/vect_embeddings.csv', 'a', newline='') as f_object:
            dictwriter_object = csv.DictWriter(f_object, fieldnames=['text', 'vect_embedding'])
            vect_emb = result[0]['vect_embeddings'].tolist()
            dictwriter_object.writerow({'text': prompt[0], 'vect_embedding': vect_emb})

        print(f'{iter_num}/{prompts_size} completed')



if __name__ == "__main__":
    fire.Fire(main)
