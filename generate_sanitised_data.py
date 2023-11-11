import os
import glob


def read_text_files(path):
    data = {}  # Create a dictionary to store file contents with file names as keys

    # Use the glob module to get a list of all text files in the directory
    text_files = sorted(glob.glob(os.path.join(path, '*.txt')))

    # Iterate through each text file and read its contents
    for file_path in text_files:
        file_name = os.path.basename(file_path)  # Get the file name without the directory
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Extract numbers from the 6th line onward
            numbers = []
            for line in lines[4:]:  # Start from the 6th line
                line = line.strip()
                if line.isdigit() and 0 <= int(line) <= 4096:
                    if file_name.startswith("Sappho_"):
                        flipped_number = 4096 - int(line)  # Flips the data from Sappho_XXXXX.txt signals
                        numbers.append(flipped_number)
                    else:
                        numbers.append(int(line))

            data[file_name] = numbers  # Store extracted numbers in the dictionary with file name as key

    return data


def split_data_into_arrays(data):
    split_arrays = {}  # Create a dictionary to store split arrays

    for file_name, file_data in data.items():
        array_size = 1500
        num_arrays = len(file_data) // array_size

        split_arrays[file_name] = []

        for i in range(num_arrays):
            start_idx = i * array_size
            end_idx = (i + 1) * array_size
            split_array = file_data[start_idx:end_idx]
            split_arrays[file_name].append(split_array)

    return split_arrays


def calculate_element_averages(split_arrays):
    element_averages = {}  # Create a dictionary to store the element averages per file

    for file_name, sublists in split_arrays.items():
        num_elements = len(sublists[0])  # Assuming all sublists have the same length

        # Initialize a list for each element's average
        averages_per_element = [0] * num_elements

        # Calculate the average for each element across all sublists
        for sublist in sublists:
            for i, value in enumerate(sublist):
                averages_per_element[i] += value

        # Divide the summed values by the number of sublists to get the average
        averages_per_element = [avg / len(sublists) for avg in averages_per_element]

        element_averages[file_name] = averages_per_element

    return element_averages


def moving_average_filter(arr):
    window_size = 3
    moving_averages = []

    for i in range(len(arr)):
        if i < window_size - 1:
            moving_averages.append(arr[i])
        else:
            window = arr[i - window_size + 1:i + 1]
            average = sum(window) / window_size
            moving_averages.append(average)

    return moving_averages


def write_averages_to_file(element_averages, output_file="sanitised_data.txt"):
    with open(output_file, 'w') as file:
        for average_list in element_averages.values():
            # Filter the avarage list
            #average_list = moving_average_filter(average_list)

            # Write each average list as a string
            file.write(str(average_list) + "\n")


directory_path = "./Samples"  # Replace with actual path
file_contents = read_text_files(directory_path)
split_data = split_data_into_arrays(file_contents)
averages = calculate_element_averages(split_data)
write_averages_to_file(averages)
