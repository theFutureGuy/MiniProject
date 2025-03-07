import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Move tensor to GPU
    device = torch.device("cuda")
    tensor = torch.randn(3, 3).to(device)
    print("Tensor on GPU:", tensor)

    # Create a simple model
    model = torch.nn.Linear(3, 1).to(device)
    print("Model on GPU:", model)
else:
    print("CUDA is not available. Running on CPU.")
