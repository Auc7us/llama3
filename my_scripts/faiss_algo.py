import torch
import numpy as np
import faiss

# Load the quantized model weights
model_path = 'path/to/your/quantized_model.pth'
model = torch.load(model_path)

def extract_2d_clusters(model):
    clusters = []
    for name, param in model.named_parameters():
        if len(param.shape) == 2:  # Ensure it's a 2D tensor
            clusters.append(param.cpu().detach().numpy())
    return clusters

clusters = extract_2d_clusters(model)

def break_into_subclusters(matrix, subcluster_size):
    d = matrix.shape[0]
    assert d % subcluster_size == 0, "Matrix size must be divisible by subcluster size"
    subclusters = (
        matrix.reshape(d // subcluster_size, subcluster_size, -1, subcluster_size)
        .swapaxes(1, 2)
        .reshape(-1, subcluster_size, subcluster_size)
    )
    return subclusters

subcluster_size = 2  
all_subclusters = []
for cluster in clusters:
    subclusters = break_into_subclusters(cluster, subcluster_size)
    all_subclusters.extend(subclusters)

flattened_subclusters = [subcluster.flatten() for subcluster in all_subclusters]
flattened_subclusters = np.array(flattened_subclusters).astype('float32')


d = flattened_subclusters.shape[1]  

res = faiss.StandardGpuResources() 
index_cpu = faiss.IndexFlatL2(d) 
index = faiss.index_cpu_to_gpu(res, 0, index_cpu) 

index.add(flattened_subclusters)

def find_similar_subclusters(index, subclusters, threshold=0.9):
    n_subclusters = subclusters.shape[0]
    similarities = []
    for i in range(n_subclusters):
        D, I = index.search(subclusters[i:i+1], n_subclusters)  
        similar = []
        for j in range(n_subclusters):
            if D[0][j] <= (1 - threshold) * d:  
                similar.append(I[0][j])
        similarities.append(similar)
    return similarities

similar_subclusters = find_similar_subclusters(index, flattened_subclusters)

# Print the result
print("Similar sub-clusters groups:")
for i, group in enumerate(similar_subclusters):
    print(f"Sub-cluster {i}: {group}")
