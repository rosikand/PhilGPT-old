"""
Need to split the .txt files since they are too big for OpenAI's limits. 
"""

import pdb

# strsight from chatgpt 
def split_file(input_file, lines_per_file):
    # Open the input file for reading
    with open(input_file, 'r') as f:
        # Read the input file and split it into lines
        lines = f.readlines()

        # Determine the number of chunks to split the file into
        num_chunks = len(lines) // lines_per_file

        # Loop over the chunks and write them to separate files
        for i in range(num_chunks):
            # Generate a filename for the current chunk
            # filename = '{}-{}'.format(i, input_file)
            filename = f"split_texts/{i}-" + input_file.split("/")[-1]

            # Open the output file for writing
            with open(filename, 'w') as chunk_file:
                # Write the current chunk to the output file
                start_idx = i * lines_per_file
                end_idx = (i + 1) * lines_per_file
                chunk_file.writelines(lines[start_idx:end_idx])

        # If there's any leftover data, write it to a final chunk
        remainder = len(lines) % lines_per_file
        if remainder > 0:
            # Generate a filename for the final chunk
            # filename = '{}-{}'.format(num_chunks, input_file)
            filename = f"split_texts/final-" + input_file.split("/")[-1]

            # Open the output file for writing
            with open(filename, 'w') as chunk_file:
                # Write the remaining data to the output file
                start_idx = num_chunks * lines_per_file
                chunk_file.writelines(lines[start_idx:])


split_file("kant/metaphysics.txt", 1000)
split_file("kant/reason.txt", 1000)
split_file("nietzsche/beyond-good-and-evil.txt", 1000)
split_file("nietzsche/zarathustra.txt", 1000)
