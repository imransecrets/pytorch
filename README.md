# Custom AI Stack with PyTorch, Llama, and Kubernetes

This guide will help you set up a custom AI stack using PyTorch, Llama, and Kubernetes for seamless deployment, scaling, and management of machine learning models.

## Table of Contents
- [Setting Up Your Environment](#setting-up-your-environment)
- [Developing Your AI Model with PyTorch](#developing-your-ai-model-with-pytorch)
- [Containerizing the Model with Docker](#containerizing-the-model-with-docker)
- [Deploying on Kubernetes](#deploying-on-kubernetes)
  - [Create a Kubernetes Cluster](#create-a-kubernetes-cluster)
  - [Create Kubernetes Manifests](#create-kubernetes-manifests)
  - [Expose the Service](#expose-the-service)
- [Integrating Llama for Enhanced Capabilities](#integrating-llama-for-enhanced-capabilities)
- [Monitoring and Scaling](#monitoring-and-scaling)
- [Conclusion](#conclusion)

## Setting Up Your Environment

Ensure you have the necessary tools and software installed on your machine:
- Docker
- Kubernetes (Minikube for local development)
- Helm
- Python 3.x
- PyTorch
- Llama (LlamaIndex or similar library for language models)

## Developing Your AI Model with PyTorch

First, create and train your AI model using PyTorch. Here’s a simple example of training a neural network:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
```
## Containerizing the Model with Docker
Next, containerize the model using Docker. Create a Dockerfile for your application:
```Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "serve.py"]
```
requirements.txt should include PyTorch and any other dependencies.

Create a serve.py script to handle inference requests:
```python
import torch
from flask import Flask, request, jsonify
from model import SimpleNN  # Your model definition

app = Flask(__name__)
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = torch.tensor(data['inputs'])
    with torch.no_grad():
        outputs = model(inputs)
    return jsonify({'outputs': outputs.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
## Deploying on Kubernetes
### Create a Kubernetes Cluster
For local development, you can use Minikube:
```bash
minikube start
```
### Create Kubernetes Manifests
Create a deployment and service for your application. Save the following as deployment.yaml:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-model
  template:
    metadata:
      labels:
        app: ai-model
    spec:
      containers:
      - name: ai-model
        image: your-docker-image:latest
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: ai-model-service
spec:
  selector:
    app: ai-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```



