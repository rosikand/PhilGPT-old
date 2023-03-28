"""
File: backend.py
------------------
Contains the logic of the GPT bot. 
"""


import openai
import pickle
import pdb
from tqdm import tqdm
import numpy as np
from transformers import GPT2TokenizerFast
import re 


TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
API_KEY = None # 'ENTER YOURS HERE!'
COMPLETIONS_MODEL = "text-davinci-003"
CHAT_GPT_MODEL = 'gpt-3.5-turbo'
EMBEDDING_MODEL = "text-embedding-ada-002"

assert API_KEY is not None, "Must provide OpenAI API Key."

openai.api_key = API_KEY


def embed(text_to_embed):
    """
    Embeds given text using OpenAI. 
    """
    result = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text_to_embed
    )

    vector = result["data"][0]["embedding"]
    return vector


def embed_content(text_files, save=True, file_path=None):
    """
    Takes in a list of text files and returns a dictionary of "chunked" embeddings. 
    Dict is of form {chunk_text: embedding}.
    """

    chapters = []
    for text_file in text_files:
        # open
        with open(text_file, "r") as f:
            text = f.read()
        
        # split into chapters
        text = re.sub(r'\n', ' ', text)
        chapters.append(text)


 
    all_embeddings = {}
    for chapter in tqdm(chapters):
        if len(chapter.split()) > 1000:
            # split into chunks 
            chunks = [chapter[i:i+1000] for i in range(0, len(chapter), 1000)]
        else:
            chunks = [chapter]

        for chunk in chunks:
            all_embeddings[chunk] = embed(chunk)


    if save:
        if file_path is None:
            with open("chapter-embeddings.pkl", "wb") as f:
                pickle.dump(all_embeddings, f)
            print("Saved chapter embeddings to chapter-embeddings.pkl")
        else:
            with open(file_path, "wb") as f:
                pickle.dump(all_embeddings, f)
            print("Saved chapter embeddings to", file_path)
        
    return all_embeddings



def vector_similarity(x, y):
    # check if numpy arrays
    assert type(x) == list or type(x) == np.ndarray
    assert type(y) == list or type(y) == np.ndarray

    if not isinstance(x, np.ndarray):
        # change to numpy arrays
        x = np.array(x)
    
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    return np.dot(np.array(x), np.array(y))



def order_document_embeddings(query, contexts):
    """
    Returns 5 document embeddings and documents with the highest similarity scores to the query.
    Returns a tuple of the form [(document, similarity_score)] in order.  
    """

    if len(query.split()) > 1000:
        raise ValueError("Query is too long for the model's max tokens. Please keep queries under 1000 words.")
    
    query_embedding = embed(query)


    # get 5 lowest similarity scores
    similarities = {}
    for text, embedding in contexts.items():
        similarity_score = vector_similarity(query_embedding, embedding)
        similarities[text] = similarity_score
    

    # sort and only keep top 5 (but also keep a map to the text)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_similarities[:5]

    return top_5


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(TOKENIZER.encode(text))


def construct_prompt(question, context_embeddings, philosopher):
    # raw gpt (no philosopher context and document texts) 
    if philosopher == "gpt":
        prompt_str = "Answer the following question as truthfully as possible \n\n Q: " + question + "\n A:"
        return  prompt_str

    SEPARATOR = "\n* "
    MAX_SECTION_LEN = 1000

    top_5_chunks = order_document_embeddings(question, context_embeddings)
    # will be of form [(document, similarity_score)]
    
    chosen_sections = []
    chosen_sections_len = 0
     
    for text_chunk, similarity_score_ in top_5_chunks:
        append_string = SEPARATOR + text_chunk.replace("\n", " ")
        chosen_sections_len += count_tokens(append_string)

        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(append_string)

    if len(chosen_sections) == 0:
        raise ValueError("No sections were chosen because each chunk was over the MAX SECTION LEN of 500 tokens.")
        

    # header = f"""Answer the question as truthfully as possible using the provided context and what you know, and if the answer is not contained within the text below, say "I don't know. Furthermore, answer the question as if you were {philosopher}, the philosopher."\n\nContext:\n"""

    # header_one = f"""Answer the question, as if you are the philosopher {philosopher}, as truthfully as possible using the provided context, which was written by {philosopher}, and what you know. If the answer is not contained within the text below, say "I don't know". \n\nContext:\n"""

    # header = f"""Answer the question, as if you are the philosopher {philosopher}, as truthfully as possible using the provided context, which was written by {philosopher}, and what you know. If you do not know, say "I don't know" or try to infer the answer based on your knowledge as a large language model and through the context provided by {philosopher}. \n\nContext:\n"""

    header = f"""Answer the question, as if you are the philosopher {philosopher}, as truthfully as possible using the provided context, which was written by {philosopher}, and what you know. \n\nContext:\n"""

    prompt_str = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
    # for debugging purposes 
    print(prompt_str)

    return prompt_str



def ask_question(question, context_embeddings, philosopher, temperature=0.0):
    """
    Asks a question using the context embeddings.
    """
    prompt = construct_prompt(question, context_embeddings, philosopher)

    response = openai.Completion.create(
        model=COMPLETIONS_MODEL,
        prompt=prompt,
        max_tokens=300,
        temperature=temperature
    )["choices"][0]["text"].strip(" \n")

    return response 
