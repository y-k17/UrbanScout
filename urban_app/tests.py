import os

def read_text(folder_path):
    file_contents = []  # This will store contents of all text files

    # Iterate over all items in the folder (both files and subfolders)
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".txt"):  # Check if the file is a text file
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        # Read the contents of the file
                        contents = file.read()
                        file_contents.append(contents)  # Append file contents to list
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    print(file_contents)

read_text("urban_app/static/Output")