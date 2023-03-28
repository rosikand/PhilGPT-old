import os
import pickle

# Path to the folder containing .pkl files
folder_path = 'nietzsche-embeddings'

# Initialize an empty list to hold the lengths of the constituent dictionaries
constituent_dict_lengths = []

# Iterate through the files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a .pkl file
    if file_name.endswith('.pkl'):
        # Load the dictionary from the file
        with open(os.path.join(folder_path, file_name), 'rb') as f:
            file_dict = pickle.load(f)
        # Append the length of the dictionary to the list
        constituent_dict_lengths.append(len(file_dict))

# Calculate the sum of the lengths of the constituent dictionaries
total_constituent_dict_length = sum(constituent_dict_lengths)

# Load the concatenated dictionary from the saved file
output_file_path = 'nietzsche-embeddings.pkl'
with open(output_file_path, 'rb') as f:
    concatenated_dict = pickle.load(f)


# Compare the length of the concatenated dictionary to the sum of the lengths of the constituent dictionaries
print('Length of concatenated dictionary:', len(concatenated_dict))
print('Sum of constituent dictionary lengths:', total_constituent_dict_length)
if len(concatenated_dict) == total_constituent_dict_length:
    print('Concatenated dictionary saved correctly')
else:
    print('Error: Concatenated dictionary has incorrect length')
