import argparse
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import OneHotEncoder

# Argument parser to receive inputs via command line
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, required=True, help="The path of input file")
parser.add_argument("-label", "--label_path", type=str, required=True, help="The path of label file")
parser.add_argument("-out", "--path_output", type=str, required=True, help="The path of output file")
parser.add_argument("-w", "--window_size", type=int, required=True, help="The window size of feature")
parser.add_argument("-dt", "--data_type", type=str, required=True, help="The data type of feature")

# Function to load input data
def loadData(path):
    with open(path, 'r') as f:
        data = [list(map(float, line.split())) for line in f]
    return np.array(data).astype('float16')

# Function to load label data
def loadLabel(path):
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
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Data Shape: {data.shape}")
    print(f"Label Shape: {label.shape}")
    np.save(os.path.join(path, "data_balanced.npy"), data)
    np.save(os.path.join(path, "label_balanced.npy"), label)

# Generate series feature with sliding window
def get_series_feature(data, window_size):
    new_dim = window_size * 2 + 1
    padded_data = np.pad(data, ((window_size, window_size), (0, 0)), mode='constant')
    result = np.zeros((data.shape[0], new_dim, data.shape[1]), dtype='float16')
    for i in range(data.shape[0]):
        start_idx = i
        end_idx = i + new_dim
        result[i] = padded_data[start_idx:end_idx]
    return result

# Select the most relevant negative samples based on cosine similarity
def select_important_negatives(positive_samples, negative_samples, num_negatives):
    neg_vec = negative_samples.reshape(negative_samples.shape[0], -1)
    pos_vec = positive_samples.reshape(positive_samples.shape[0], -1)
    similarity_scores = cosine_similarity(neg_vec, pos_vec).mean(axis=1)
    selected_indices = np.argsort(similarity_scores)[:num_negatives]
    return negative_samples[selected_indices]

# Process a single file
def process_file(input_file, label_file_path, window_size):
    data = loadData(input_file)
    labels = loadLabel(label_file_path)
    if data.shape[0] != len(labels):
        min_length = min(data.shape[0], len(labels))
        data = data[:min_length]
        labels = labels[:min_length]
    series_features = get_series_feature(data, window_size)
    positive_samples = series_features[labels == 1]
    negative_samples = series_features[labels == 0]
    return positive_samples, negative_samples

# Main function to combine all data and labels
def main(path_input, path_output, window_size, data_type, label_path):
    input_files = [f for f in os.listdir(path_input) if f.endswith(data_type)]
    label_files = os.listdir(label_path)
    positive_samples_list = []
    negative_samples_list = []
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for input_file in input_files:
            file_name = input_file.split(".")[0]
            label_file_path = os.path.join(label_path, f"{file_name}.txt")
            if not os.path.exists(label_file_path):
                print(f"Label file for {file_name} not found. Skipping...")
                continue
            futures.append(executor.submit(process_file, os.path.join(path_input, input_file), label_file_path, window_size))
        
        for future in as_completed(futures):
            pos_samples, neg_samples = future.result()
            if pos_samples is not None and neg_samples is not None:
                positive_samples_list.append(pos_samples)
                negative_samples_list.append(neg_samples)
    
    if positive_samples_list and negative_samples_list:
        positive_samples = np.concatenate(positive_samples_list, axis=0)
        negative_samples = np.concatenate(negative_samples_list, axis=0)
        num_negatives = int(len(positive_samples) * 1.2)  # Ensure 1.2 times positive samples
        selected_negative_samples = select_important_negatives(positive_samples, negative_samples, num_negatives)
        selected_negative_samples = selected_negative_samples.reshape(
            selected_negative_samples.shape[0], window_size * 2 + 1, positive_samples.shape[2]
        )
        balanced_data = np.concatenate((positive_samples, selected_negative_samples), axis=0)
        balanced_labels = np.concatenate((np.ones(len(positive_samples)), np.zeros(len(selected_negative_samples))), axis=0)
        labels_one_hot = one_hot_encode(balanced_labels)
        saveData(path_output, balanced_data, labels_one_hot)
    else:
        print("No data and labels to process. Please check your input files.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output, args.window_size, args.data_type, args.label_path)
