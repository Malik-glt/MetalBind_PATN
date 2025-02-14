#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-query", "--query_path", type=str, required=True, help="Path to the query .npy file")
parser.add_argument("-labels", "--labels_path", type=str, required=True, help="Path to the query labels .npy file")
parser.add_argument("-db", "--database_path", type=str, required=True, help="Path to the RAG database .npy file")
parser.add_argument("-out", "--output_path", type=str, required=True, help="Path to save the fused embeddings")
parser.add_argument("-batch", "--batch_size", type=int, default=100, help="Number of query sequences per batch")
args = parser.parse_args()

# Parameters
query_path = args.query_path
labels_path = args.labels_path
database_path = args.database_path
output_path = args.output_path
batch_size = args.batch_size

# Fixed parameters
maxseq = 15
num_feature = 1024  # Embedding dimension

# Load Query, Labels, and RAG Database
query_data = np.load(query_path)  # Shape: (N, 1, 15, 1024)
labels = np.load(labels_path)  # Shape: (N, 2) → One-hot encoded labels
rag_database = np.load(database_path)  # Shape: (605679, 1, 15, 1024)

# Function to find the top 5 most similar sequences and fuse embeddings
def compute_rag_embedding(query, database, maxseq, num_feature):
    """
    Finds the 5 closest sequences in the database, averages them, and fuses the result with the query embedding.
    """
    query = query.reshape(1, maxseq, num_feature)  # Reshape for consistency
    database = database.reshape(database.shape[0], maxseq, num_feature)  # Remove singleton dimension

    # Compute L2 distances efficiently
    distances = np.linalg.norm(database - query, axis=(1, 2))  # Compute distances across all sequences

    # Get indices of 5 closest sequences
    top_5_indices = np.argsort(distances)[:5]
    closest_sequences = database[top_5_indices]  # Retrieve the closest 5 embeddings

    # Compute the average of top-5 sequences
    avg_embedding = np.mean(closest_sequences, axis=0)

    # Fuse query and RAG embedding (70% query, 30% RAG)
    fused_embedding = (query * 0.7) + (avg_embedding * 0.3)

    return fused_embedding.reshape(1, maxseq, num_feature)  # Reshape to original format

# Initialize an array to store all fused embeddings (same as query_data)
fused_embeddings = np.copy(query_data)

# Process Query Data in Batches
num_queries = query_data.shape[0]
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"Total queries: {num_queries}")
print(f"Processing in {num_batches} batches of size {batch_size}")

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_queries)
    batch_queries = query_data[start_idx:end_idx]
    batch_labels = labels[start_idx:end_idx]

    for i, (query, label) in enumerate(zip(batch_queries, batch_labels)):
        if label[1] == 1:  # Apply RAG only on positive samples (one-hot encoding check)
            fused_embedding = compute_rag_embedding(query, rag_database, maxseq, num_feature)
            fused_embeddings[start_idx + i] = fused_embedding  # Update only positive samples

    print(f"Batch {batch_idx + 1}/{num_batches} processed")

# Save the final fused embeddings
np.save(output_path, fused_embeddings)
print(f"Fused embeddings saved to {output_path}")


# print("Anh bi khung")
