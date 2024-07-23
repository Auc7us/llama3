import torch
import torch.nn as nn
import faiss

# Define a simple model with the specified layer sizes
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(7, 6, bias=False)  # (6, 7)
        self.layer2 = nn.Linear(5, 7, bias=False)  # (7, 5)
        self.layer3 = nn.Linear(6, 5, bias=False)  # (5, 6)
        self.layer4 = nn.Linear(7, 7, bias=False)  # (7, 7)
        self.layer5 = nn.Linear(7, 4, bias=False)  # (4, 7)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

model = SimpleModel()

# Set the model weights
with torch.no_grad():
    model.layer1.weight = nn.Parameter(torch.tensor(layers[0].T, dtype=torch.float32))
    model.layer2.weight = nn.Parameter(torch.tensor(layers[1].T, dtype=torch.float32))
    model.layer3.weight = nn.Parameter(torch.tensor(layers[2].T, dtype=torch.float32))
    model.layer4.weight = nn.Parameter(torch.tensor(layers[3].T, dtype=torch.float32))
    model.layer5.weight = nn.Parameter(torch.tensor(layers[4].T, dtype=torch.float32))

# Save the model
model_path = 'pseudo_model.pth'
torch.save(model.state_dict(), model_path)

# Load the model weights
model = SimpleModel()
model.load_state_dict(torch.load(model_path))
model.eval()

def extract_2d_clusters(model):
    clusters = []
    for name, param in model.named_parameters():
        if len(param.shape) == 2:  # Ensure it's a 2D tensor
            clusters.append(param.cpu().detach().numpy())
    return clusters

clusters = extract_2d_clusters(model)

def break_into_subclusters(matrix, subcluster_size):
    d1, d2 = matrix.shape
    subclusters = []
    for i in range(0, d1, subcluster_size):
        for j in range(0, d2, subcluster_size):
            subcluster = matrix[i:i+subcluster_size, j:j+subcluster_size]
            if subcluster.shape == (subcluster_size, subcluster_size):
                subclusters.append(subcluster)
    return subclusters

subcluster_size = 2  # Example size of sub-clusters
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


print("Similar sub-clusters groups:")
for i, group in enumerate(similar_subclusters):
    print(f"Sub-cluster {i}: {group}")
