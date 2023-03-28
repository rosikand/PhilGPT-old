import os
import pickle

# Path to the folder containing .pkl files
folder_path = 'nietzsche-embeddings'

# Initialize an empty dictionary to hold the concatenated dictionaries
concatenated_dict = {}

# Iterate through the files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a .pkl file
    if file_name.endswith('.pkl'):
        # Load the dictionary from the file
        with open(os.path.join(folder_path, file_name), 'rb') as f:
            file_dict = pickle.load(f)
        # Update the concatenated dictionary with the loaded dictionary
        concatenated_dict.update(file_dict)


# Save the concatenated dictionary to a new .pkl file
output_file_path = 'nietzsche-embeddings.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(concatenated_dict, f)

# Print a confirmation message
print(f'Concatenated dictionary saved to {output_file_path}')
