import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input file")
parser.add_argument("-out","--path_output", type=str, help="the path of output file")
parser.add_argument("-w","--window_size", type=int, help="the window_size of series feature")


def loadData(path):
    Data = np.loadtxt(path)
    return Data

def saveData(path, data):
    np.save(path, data)

def get_series_feature(data, window_size):
    new_dim = window_size * 2 + 1
    padded_data = np.pad(data, ((window_size, window_size), (0, 0)), mode='constant')
    result = np.zeros((data.shape[0], new_dim, data.shape[1]))
    for i in range(data.shape[0]):
        start_idx = i
        end_idx = i + new_dim
        for j in range(start_idx, end_idx):
            result[i][j - start_idx] = padded_data[j]
    
    return result

def main(path_input, path_output, window_size):
    data = loadData(path_input)
    result = get_series_feature(data, window_size)
    saveData(path_output, result)

if __name__ == "__main__":
    args = parser.parse_args()
    path_input = args.path_input
    path_output = args.path_output
    window_size = args.window_size
    main(path_input, path_output, window_size)