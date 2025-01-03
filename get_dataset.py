import argparse
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# Argument parser to receive inputs via command line
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="The path of input file")
parser.add_argument("-label", "--label_path", type=str, help="The path of label file")
parser.add_argument("-out", "--path_output", type=str, help="The path of output file")
parser.add_argument("-w", "--window_size", type=int, help="The window size of feature")
parser.add_argument("-dt", "--data_type", type=str, help="The data type of feature")

# Function to load input data
def loadData(path):
    with open(path, 'r') as f:
        data = [list(map(float, line.split())) for line in f]
    return np.array(data).astype('float16')

# Function to load label data
def loadlabel(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    y_data = np.array([int(x) for x in lines[1].strip()])  # Skip header
    return y_data.astype('float16')

# One-hot encoding for labels
def one_hot_encode(labels):
    encoder = OneHotEncoder(sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot_labels = encoder.fit_transform(labels)
    return one_hot_labels.astype('float16')

# Save data and labels efficiently
def saveData(path, data, label):
    try:
        # Ensure output directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Print shapes to confirm before saving
        print(f"Data Shape: {data.shape}")
        print(f"Label Shape: {label.shape}")
        
        # Save in smaller chunks if dataset is large
        np.save(os.path.join(path, "data.npy"), data)
        np.save(os.path.join(path, "label.npy"), label)

    except OSError as e:
        print(f"Error saving data: {e}")
        print("Attempting to save in smaller chunks...")
        
        # Save in smaller parts to avoid memory issues
        chunk_size = 10000
        for i in range(0, data.shape[0], chunk_size):
            np.save(os.path.join(path, f"data_part_{i}.npy"), data[i:i + chunk_size])
            np.save(os.path.join(path, f"label_part_{i}.npy"), label[i:i + chunk_size])

# Generate series feature with sliding window
def get_series_feature(data, window_size):
    new_dim = window_size * 2 + 1
    # Pad the data to handle edges
    padded_data = np.pad(data, ((window_size, window_size), (0, 0)), mode='constant')
    result = np.zeros((data.shape[0], new_dim, data.shape[1]), dtype='float16')

    # Generate features using sliding window
    for i in range(data.shape[0]):
        start_idx = i
        end_idx = i + new_dim
        result[i] = padded_data[start_idx:end_idx]
    
    return result

# Main function to combine all data and labels
def main(path_input, path_output, window_size, data_type, label_path):
    result = []
    label = []

    # Iterate over input files and process each one
    input_files = os.listdir(path_input)
    for i in input_files:
        if i.endswith(data_type):
            file_name = i.split(".")[0]
            
            # Load data and labels
            data = loadData(os.path.join(path_input, i))
            result.append(get_series_feature(data, window_size))
            label.append(loadlabel(os.path.join(label_path, f"{file_name}.label")))

    # Combine all results and labels
    data = np.concatenate(result, axis=0)
    labels = np.concatenate(label, axis=0)
    
    # Expand data dimensions to match (N, 1, 15, 768) shape
    data = np.expand_dims(data, axis=1)

    # Convert labels to one-hot encoding
    labels_one_hot = one_hot_encode(labels)

    # Save the combined data and labels
    saveData(path_output, data, labels_one_hot)

# Entry point
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Retrieve command-line arguments
    path_input = args.path_input
    path_output = args.path_output
    window_size = args.window_size
    data_type = args.data_type
    label_path = args.label_path

    # Run main function
    main(path_input, path_output, window_size, data_type, label_path)
