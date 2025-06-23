#!/bin/bash

echo "🚀 Starting EKS deployment for CodeCraft AI Assistant..."

# Step 1: Create EKS cluster
echo "📦 Creating EKS cluster..."
eksctl create cluster -f eks-cluster.yaml

# Step 2: Update kubeconfig
echo "🔧 Updating kubeconfig..."
aws eks update-kubeconfig --region us-east-1 --name codecraft-cluster

# Step 3: Install AWS Load Balancer Controller
echo "⚖️ Installing AWS Load Balancer Controller..."
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=codecraft-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Step 4: Build and push Docker image
echo "🐳 Building and pushing Docker image..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 149536455493.dkr.ecr.us-east-1.amazonaws.com
docker build -t codecraft .
docker tag codecraft:latest 149536455493.dkr.ecr.us-east-1.amazonaws.com/codecraft:latest
docker push 149536455493.dkr.ecr.us-east-1.amazonaws.com/codecraft:latest

# Step 5: Deploy application
echo "🚀 Deploying CodeCraft application..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Step 6: Check deployment status
echo "📊 Checking deployment status..."
kubectl get pods -n codecraft
kubectl get svc -n codecraft
kubectl get ingress -n codecraft

echo "✅ Deployment complete! Check AWS Console for Load Balancer URL"