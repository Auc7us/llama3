import numpy as np
import faiss

d = 128
nb = 1000
nq = 5

np.random.seed(1234)  
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.0  
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.0

res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(d)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

print(f"Is trained: {gpu_index_flat.is_trained}")
print(f"Add vectors to the index")

gpu_index_flat.add(xb)  
print(f"Number of vectors in the index: {gpu_index_flat.ntotal}")

k = 6
D, I = gpu_index_flat.search(xq, k) 

print("Query vectors (xq):")
print(xq)

print("\nNearest neighbors' indices (I):")
print(I)

print("\nNearest neighbors' distances (D):")
print(D)
