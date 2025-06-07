#!/usr/bin/env python
# coding: utf-8
import numpy as np
import argparse
import time

# Try to import GPU libraries
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("PyTorch with CUDA support detected")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, falling back to CPU")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy detected for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-query", "--query_path", type=str, required=True, help="Path to the query .npy file")
parser.add_argument("-labels", "--labels_path", type=str, required=False, help="Path to the query labels .npy file (optional, for statistics only)")
parser.add_argument("-db", "--database_path", type=str, required=True, help="Path to the RAG database .npy file")
parser.add_argument("-out", "--output_path", type=str, required=True, help="Path to save the fused embeddings")
parser.add_argument("-batch", "--batch_size", type=int, default=100, help="Number of query sequences per batch")
parser.add_argument("-gpu", "--use_gpu", action="store_true", help="Use GPU acceleration if available")
parser.add_argument("-backend", "--gpu_backend", type=str, choices=["torch", "cupy", "auto"], default="auto", 
                   help="GPU backend to use (torch/cupy/auto)")
args = parser.parse_args()

# Parameters
query_path = args.query_path
labels_path = args.labels_path
database_path = args.database_path
output_path = args.output_path
batch_size = args.batch_size
use_gpu = args.use_gpu
gpu_backend = args.gpu_backend

# Fixed parameters
maxseq = 15
num_feature = 1024  # Embedding dimension

# Determine GPU backend
import torch
import importlib

# Flags for backend availability
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and torch.cuda.is_available()
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None

# User options
use_gpu = True
gpu_backend = "auto"  # Can be "auto", "torch", or "cupy"

# GPU backend selection
if use_gpu:
    if gpu_backend == "auto":
        if TORCH_AVAILABLE:
            gpu_backend = "torch"
            print(f"✅ Auto-selected PyTorch backend with GPU: {torch.cuda.get_device_name(0)}")
        elif CUPY_AVAILABLE:
            gpu_backend = "cupy"
            import cupy
            print("✅ Auto-selected CuPy backend")
        else:
            print("❌ No GPU backend available, falling back to CPU")
            use_gpu = False
    elif gpu_backend == "torch":
        if not TORCH_AVAILABLE:
            print("❌ PyTorch with CUDA not available, falling back to CPU")
            use_gpu = False
    elif gpu_backend == "cupy":
        if not CUPY_AVAILABLE:
            print("❌ CuPy not available, falling back to CPU")
            use_gpu = False

# Final status
print(f"Using GPU: {use_gpu}")
if use_gpu:
    print(f"GPU Backend: {gpu_backend}")


# Load Query and RAG Database
print("Loading data...")
query_data = np.load(query_path)  # Shape: (N, 1, 15, 1024)
rag_database = np.load(database_path)  # Shape: (605679, 1, 15, 1024)

# Load labels if provided (for statistics only, not used in processing)
labels = None
if labels_path:
    labels = np.load(labels_path)  # Shape: (N, 2) → One-hot encoded labels

print(f"Query data shape: {query_data.shape}")
print(f"Database shape: {rag_database.shape}")

def compute_rag_embedding_torch(query_batch, database, maxseq, num_feature, device):
    """
    GPU-accelerated RAG embedding computation using PyTorch.
    Fixed to process each query individually to avoid memory overflow.
    """
    # Move database to GPU once
    database_gpu = torch.from_numpy(database).float().to(device)
    database_gpu = database_gpu.squeeze(1)  # (M, 15, 1024)
    
    batch_size_actual = query_batch.shape[0]
    batch_results = []
    
    # Process each query in the batch individually
    for i in range(batch_size_actual):
        # Move single query to GPU
        query = torch.from_numpy(query_batch[i:i+1]).float().to(device)  # (1, 1, 15, 1024)
        query = query.squeeze(1)  # (1, 15, 1024)
        
        # Compute L2 distances for this single query
        # query: (1, 15, 1024), database_gpu: (M, 15, 1024)
        # Expand query to match database dimensions for broadcasting
        query_expanded = query.expand(database_gpu.shape[0], -1, -1)  # (M, 15, 1024)
        
        # Compute distances: (M,)
        distances = torch.norm(database_gpu - query_expanded, dim=(1, 2))
        
        # Get top-5 indices
        top_5_indices = torch.topk(distances, k=5, largest=False).indices  # (5,)
        
        # Get closest sequences
        closest_sequences = database_gpu[top_5_indices]  # (5, 15, 1024)
        avg_embedding = torch.mean(closest_sequences, dim=0, keepdim=True)  # (1, 15, 1024)
        
        # Fuse query and RAG embedding (70% query, 30% RAG)
        fused_embedding = (query * 0.7) + (avg_embedding * 0.3)
        batch_results.append(fused_embedding)
    
    # Stack results and move back to CPU
    fused_batch = torch.cat(batch_results, dim=0)  # (batch_size, 15, 1024)
    fused_batch = fused_batch.unsqueeze(1)  # (batch_size, 1, 15, 1024)
    
    return fused_batch.cpu().numpy()

def compute_rag_embedding_torch_chunked(query_batch, database_gpu, maxseq, num_feature, device, chunk_size=1000):
    """
    Alternative memory-efficient version that processes database in chunks.
    Use this if even single queries cause memory issues with large databases.
    """
    batch_size_actual = query_batch.shape[0]
    batch_results = []
    db_size = database_gpu.shape[0]
    
    for i in range(batch_size_actual):
        query = torch.from_numpy(query_batch[i:i+1]).float().to(device)
        query = query.squeeze(1)  # (1, 15, 1024)
        
        min_distances = None
        min_indices = None
        
        # Process database in chunks
        for chunk_start in range(0, db_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, db_size)
            db_chunk = database_gpu[chunk_start:chunk_end]  # (chunk_size, 15, 1024)
            
            # Compute distances for this chunk
            query_expanded = query.expand(db_chunk.shape[0], -1, -1)
            distances = torch.norm(db_chunk - query_expanded, dim=(1, 2))
            
            # Keep track of minimum distances
            if min_distances is None:
                min_distances = distances
                min_indices = torch.arange(chunk_start, chunk_end, device=device)
            else:
                # Combine with existing minimums
                combined_distances = torch.cat([min_distances, distances])
                combined_indices = torch.cat([min_indices, torch.arange(chunk_start, chunk_end, device=device)])
                
                # Keep only top-5
                top_5_vals, top_5_pos = torch.topk(combined_distances, k=min(5, len(combined_distances)), largest=False)
                min_distances = top_5_vals
                min_indices = combined_indices[top_5_pos]
        
        # Get final top-5 sequences
        closest_sequences = database_gpu[min_indices]  # (5, 15, 1024)
        avg_embedding = torch.mean(closest_sequences, dim=0, keepdim=True)  # (1, 15, 1024)
        
        # Fuse embeddings
        fused_embedding = (query * 0.7) + (avg_embedding * 0.3)
        batch_results.append(fused_embedding)
    
    fused_batch = torch.cat(batch_results, dim=0)
    fused_batch = fused_batch.unsqueeze(1)
    
    return fused_batch.cpu().numpy()

def compute_rag_embedding_cupy(query_batch, database, maxseq, num_feature):
    """
    GPU-accelerated RAG embedding computation using CuPy.
    """
    # Move data to GPU
    query_batch_gpu = cp.asarray(query_batch)  # (batch_size, 1, 15, 1024)
    database_gpu = cp.asarray(database)  # (M, 1, 15, 1024)
    
    # Reshape for processing
    query_batch_gpu = query_batch_gpu.squeeze(1)  # (batch_size, 15, 1024)
    database_gpu = database_gpu.squeeze(1)  # (M, 15, 1024)
    
    batch_size_actual = query_batch_gpu.shape[0]
    
    batch_results = []
    for i in range(batch_size_actual):
        query = query_batch_gpu[i:i+1]  # (1, 15, 1024)
        
        # Compute L2 distances
        distances = cp.linalg.norm(database_gpu - query, axis=(1, 2))
        
        # Get top-5 indices
        top_5_indices = cp.argsort(distances)[:5]
        closest_sequences = database_gpu[top_5_indices]  # (5, 15, 1024)
        
        # Compute average
        avg_embedding = cp.mean(closest_sequences, axis=0, keepdims=True)  # (1, 15, 1024)
        
        # Fuse embeddings
        fused_embedding = (query * 0.7) + (avg_embedding * 0.3)
        batch_results.append(fused_embedding)
    
    # Stack and reshape results
    fused_batch = cp.stack(batch_results, axis=0)  # (batch_size, 1, 15, 1024)
    fused_batch = fused_batch.reshape(batch_size_actual, 1, maxseq, num_feature)
    
    # Move back to CPU
    return cp.asnumpy(fused_batch)

def compute_rag_embedding_cpu(query, database, maxseq, num_feature):
    """
    Original CPU implementation for fallback.
    """
    query = query.reshape(1, maxseq, num_feature)
    database = database.reshape(database.shape[0], maxseq, num_feature)
    
    # Compute L2 distances efficiently
    distances = np.linalg.norm(database - query, axis=(1, 2))
    
    # Get indices of 5 closest sequences
    top_5_indices = np.argsort(distances)[:5]
    closest_sequences = database[top_5_indices]
    
    # Compute the average of top-5 sequences
    avg_embedding = np.mean(closest_sequences, axis=0)
    
    # Fuse query and RAG embedding (70% query, 30% RAG)
    fused_embedding = (query * 0.7) + (avg_embedding * 0.3)
    
    return fused_embedding.reshape(1, maxseq, num_feature)

# Initialize an array to store all fused embeddings
fused_embeddings = np.copy(query_data)

# Process Query Data in Batches
num_queries = query_data.shape[0]
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"Total queries: {num_queries}")
print(f"Processing in {num_batches} batches of size {batch_size}")
print("Applying RAG to ALL samples uniformly (no label-dependent processing)")

# Setup GPU device if using PyTorch
device = None
database_gpu = None
if use_gpu and gpu_backend == "torch":
    device = torch.device("cuda")
    # Pre-load database to GPU for PyTorch (if memory allows)
    try:
        database_gpu = torch.from_numpy(rag_database).float().to(device)
        database_gpu = database_gpu.squeeze(1)  # (M, 15, 1024)
        print("Database preloaded to GPU memory")
        preloaded_db = True
    except RuntimeError as e:
        print(f"Cannot preload database to GPU: {e}")
        print("Will use chunked processing or CPU fallback")
        preloaded_db = False

start_time = time.time()

for batch_idx in range(num_batches):
    batch_start_time = time.time()
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_queries)
    
    batch_queries = query_data[start_idx:end_idx]
    
    if use_gpu:
        if gpu_backend == "torch":
            if preloaded_db:
                try:
                    # Use the fixed function with per-query processing
                    fused_batch = compute_rag_embedding_torch(batch_queries, rag_database, maxseq, num_feature, device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"GPU memory error in batch {batch_idx + 1}, trying chunked approach...")
                        try:
                            fused_batch = compute_rag_embedding_torch_chunked(batch_queries, database_gpu, maxseq, num_feature, device)
                        except RuntimeError as e2:
                            print(f"Chunked approach also failed, falling back to CPU for this batch: {e2}")
                            # CPU fallback for this batch
                            fused_batch = np.zeros_like(batch_queries)
                            for i, query in enumerate(batch_queries):
                                fused_embedding = compute_rag_embedding_cpu(query, rag_database, maxseq, num_feature)
                                fused_batch[i] = fused_embedding
                    else:
                        raise e
            else:
                # CPU fallback
                fused_batch = np.zeros_like(batch_queries)
                for i, query in enumerate(batch_queries):
                    fused_embedding = compute_rag_embedding_cpu(query, rag_database, maxseq, num_feature)
                    fused_batch[i] = fused_embedding
        elif gpu_backend == "cupy":
            fused_batch = compute_rag_embedding_cupy(batch_queries, rag_database, maxseq, num_feature)
        
        # Update embeddings for the batch
        fused_embeddings[start_idx:end_idx] = fused_batch
    else:
        # CPU processing
        for i, query in enumerate(batch_queries):
            fused_embedding = compute_rag_embedding_cpu(query, rag_database, maxseq, num_feature)
            fused_embeddings[start_idx + i] = fused_embedding
    
    batch_time = time.time() - batch_start_time
    print(f"Batch {batch_idx + 1}/{num_batches} processed ({batch_time:.2f}s)")

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f}s")
print(f"Average time per query: {total_time/num_queries:.4f}s")

# Save the final fused embeddings
np.save(output_path, fused_embeddings)
print(f"Fused embeddings saved to {output_path}")

# Print label distribution for verification (if labels provided)
if labels is not None:
    pos_count = np.sum(labels[:, 1])
    neg_count = np.sum(labels[:, 0])
    print(f"Label distribution: {pos_count} positive, {neg_count} negative samples")
print("RAG applied uniformly to all samples - no data leakage")