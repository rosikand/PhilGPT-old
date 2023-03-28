"""
File: process_text.py
------------------
Some Plato texts preprocessing operations.   
"""


import backend
import pdb
import sys
import time

filestr = sys.argv[1]

text_files = [f"split_texts/{filestr}.txt"]

print(text_files)


curr_embedding = backend.embed_content(text_files, save=True, file_path = f'nietzsche-embeddings/{filestr}.pkl') 

print("Done! now sleeping...")
time.sleep(61)
print("-----------")
print("-----------")
print("-----------")
print("-----------")
print("-----------")
print("-----------")

