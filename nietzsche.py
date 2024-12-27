import os

# specify folder containing text files and the output file path
folder_path = "Nietzsche"
output_file = os.path.join("train_data", "all_nietzsche.txt")  # specify the output file name

# open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # loop through all files in the folder
    for file_name in os.listdir(folder_path):
        # process only .txt files
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8-sig') as infile:
                # write content of each file to the output file
                outfile.write(infile.read())
                outfile.write("\n")  # optionally, add a newline to separate file contents
