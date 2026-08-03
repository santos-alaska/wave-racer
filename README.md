# Wave Racer — CI/CD Pipeline on AWS EKS

A production-style CI/CD pipeline for **Wave Racer**, a WebGL game built with Three.js. This project demonstrates a full GitOps workflow: developers push code to GitHub, Jenkins handles Continuous Integration (build & push Docker images), and ArgoCD handles Continuous Deployment to a Kubernetes cluster on AWS EKS — with Prometheus & Grafana for monitoring.

---

## 📐 Architecture / Pipeline Flow

1. A developer writes code and pushes it to the `main` branch on GitHub.
2. Once code lands on `main`, the pipeline is triggered automatically.
3. **Jenkins** performs the CI activity:
   - Pulls the latest code from GitHub
   - Builds a Docker image
   - Tags the image with a unique build number
   - Pushes the image to Docker Hub
   - Updates the image tag in the Kubernetes deployment manifest and pushes that change back to GitHub
4. **ArgoCD** performs the CD activity:
   - Detects the manifest change in GitHub (GitOps)
   - Automatically syncs and deploys the new version into the EKS cluster
5. The new version of the app becomes live, accessible via a Kubernetes `LoadBalancer` Service.

**For new features:**
- Developer creates a feature branch from `main`
- Adds changes, pushes the branch, and opens a Pull Request
- A reviewer reviews and merges the PR into `main`
- The merge triggers the pipeline automatically → Jenkins builds → ArgoCD deploys → new feature goes live

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Source Control | GitHub |
| CI | Jenkins |
| Containerization | Docker / Docker Hub |
| Orchestration | Kubernetes (AWS EKS) |
| CD / GitOps | ArgoCD |
| Monitoring | Prometheus + Grafana |
| IaC / Cluster Provisioning | eksctl |

---

## 🏗️ Project Resources

| Resource | Name |
|---|---|
| EC2 Instance (Jenkins/CI server) | `Wave-racer-server` |
| EKS Cluster | `wave-racer` |
| Docker Hub Repo | `nwachukwuchukwuka123/wave-racer` |
| GitHub Repo | `santos-alaska/wave-racer` |
| ArgoCD Application | `wave-racer` |
| Kubernetes Deployment | `wave-racer-app` |
| Kubernetes Service | `wave-racer-service` |
| Namespace | `default` |

---

## ⚙️ Setup Guide

### 1. Launch the CI/CD Server
- Launch an EC2 instance:
  - AMI: **Ubuntu 24.04**
  - Instance type: **t2.large**
  - Storage: **30 GB**
  - Name: `Wave-racer-server`
- Connect via SSH and switch to root, then update packages.

### 2. Install Jenkins
```bash
sudo apt install openjdk-17-jre-headless -y

sudo wget -O /usr/share/keyrings/jenkins-keyring.asc \
  https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

sudo apt-get update
sudo apt-get install jenkins -y
```

### 3. Install Docker
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
docker --version
```

### 4. Install kubectl, AWS CLI, and eksctl
```bash
# kubectl
curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.19.6/2021-01-05/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin
kubectl version --client

# AWS CLI
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version

# eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
eksctl version
```

### 5. Configure IAM Access for the EC2 Instance
- Create an IAM Role:
  - Trusted entity: **AWS service**
  - Use case: **EC2**
  - Permissions: **AdministratorAccess**
  - Role name: `production-grade-app`
- Attach the role to the `Wave-racer-server` EC2 instance (**Actions → Security → Modify IAM role**).

### 6. Create the EKS Cluster
```bash
eksctl create cluster --name wave-racer --region us-east-1 --node-type t2.medium --zones us-east-1a,us-east-1b
```

### 7. Configure Jenkins Plugins
Install the following plugins (**Manage Jenkins → Plugins**):
- Pipeline: Stage View
- Docker
- Docker Commons
- Docker Pipeline
- Docker API
- Docker build step
- Kubernetes
- Kubernetes Client API
- Credentials
- Credentials Binding
- CLI

Restart Jenkins after installing.

### 8. Configure Jenkins Tools & Credentials

**Docker tool:**
- Manage Jenkins → Tools → Add Docker → Name: `docker` (no installation needed)

**Docker Hub credentials:**
- Manage Jenkins → Credentials → Global → Add Credentials
- Kind: Username and password
- Username: your Docker Hub username
- Password: Docker Hub personal access token
- ID: `dockerhub-creds`

**GitHub credentials:**
- GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
- Select all scopes → Generate → copy the token
- Manage Jenkins → Credentials → Add Credentials
- Kind: Username and password
- Username: your GitHub username
- Password: the generated token
- ID: `github-creds`

### 9. Push the Application Code
- Create the GitHub repo `wave-racer`
- Add the application source, `Dockerfile`, `k8s/` manifests, and `Jenkinsfile`
- Push to `main`

### 10. Configure kubectl Access for the Jenkins User

> **Note:** Unlike copying root's kubeconfig over to the `jenkins` user, this project configured AWS/Kubernetes access **directly as the `jenkins` user** — avoiding the need to `mkdir`/`cp`/`chown` root's kubeconfig into `/home/jenkins/.kube/`.

```bash
# Add jenkins to the docker group so it can run docker commands
sudo usermod -aG docker jenkins

# Switch to the jenkins user
sudo su - jenkins

# Configure AWS credentials for the jenkins user
aws configure

# Verify identity
aws sts get-caller-identity

# Generate a kubeconfig for the jenkins user, pointing at the wave-racer cluster
aws eks update-kubeconfig --region us-east-1 --name wave-racer

# Exit back to root
exit

# Restart Jenkins to pick up group/environment changes
sudo systemctl restart jenkins
```

> If the IAM identity used by `jenkins` differs from the one that created the cluster, you may also need to grant it access:
> ```bash
> eksctl create iamidentitymapping \
>   --cluster wave-racer \
>   --region us-east-1 \
>   --arn arn:aws:iam::<ACCOUNT_ID>:user/<jenkins-iam-user> \
>   --group system:masters \
>   --username jenkins
> ```

### 11. Install ArgoCD (via Helm)
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version

helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

kubectl create namespace argocd
helm install argocd argo/argo-cd -n argocd

kubectl get pods -n argocd
kubectl get svc -n argocd
```

Expose ArgoCD via LoadBalancer:
```bash
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
kubectl get svc argocd-server -n argocd
```
Copy the `EXTERNAL-IP` and open it in your browser: `https://<EXTERNAL-IP>`

Get the ArgoCD admin password:
```bash
kubectl get secret argocd-initial-admin-secret \
  -n argocd \
  -o jsonpath="{.data.password}" | base64 -d
```
- Username: `admin`
- Password: (output above)

### 12. Create the Jenkins Pipeline
- Jenkins → New Item → **Multibranch Pipeline**
- Display name: `Wave-Racer-App`
- Description: *Production-grade deployment pipeline for Wave Racer*
- Branch Source: Git → paste the repo URL
- Property strategy: Named branches get specific properties
- **Do not save yet.**

### 13. Create the ArgoCD Application
- ArgoCD UI → **New App**
- App name: `wave-racer`
- Project: `default`
- Sync policy: **Automatic** (enable Prune + Self Heal)
- Source:
  - Repository URL: `https://github.com/santos-alaska/wave-racer.git`
  - Revision: `main`
  - Path: `k8s`
- Destination:
  - Cluster URL: `https://kubernetes.default.svc`
  - Namespace: `default`
- **Create**

### 14. Finish Jenkins Setup
- Go back to Jenkins → **Apply and Save**
- It should trigger a scan/build automatically
- Re-open the pipeline project to view the build

### 15. Verify the Deployment
```bash
kubectl get pods -n default
kubectl get svc -n default
kubectl get all -n default
```
Copy the `EXTERNAL-IP` of `wave-racer-service` and open it in a browser to see the live app — this is **v1** of the project.

---

## 📊 Monitoring — Prometheus & Grafana

### Install (via Helm)
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create namespace monitoring

helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring
```

Verify all components are running:
```bash
kubectl get pods -n monitoring
```
Expect to see: `prometheus`, `alertmanager`, `grafana`, `node-exporter`, `kube-state-metrics`.

### Access Grafana
```bash
kubectl patch svc monitoring-grafana -n monitoring -p '{"spec":{"type":"LoadBalancer"}}'
kubectl get svc monitoring-grafana -n monitoring
```
Open `http://<EXTERNAL-IP>` in your browser.

Get the Grafana admin password:
```bash
kubectl get secret monitoring-grafana \
  -n monitoring \
  -o jsonpath="{.data.admin-password}" | base64 -d
```
- Username: `admin`
- Password: (output above)

### Dashboards
The `kube-prometheus-stack` chart auto-installs several useful dashboards — no manual import needed:
- Kubernetes / Nodes
- Kubernetes / Pods
- Kubernetes / Deployments
- Kubernetes / Cluster
- Node Exporter Full

---

## 🔁 Development Workflow (Shipping a New Feature)

```bash
# Create a feature branch from main
git checkout main
git pull origin main
git checkout -b featureA

# Make changes, then push
git add .
git commit -m "Add featureA"
git push origin featureA
```

- Open a Pull Request on GitHub (`featureA` → `main`)
- Reviewer reviews and **merges** the PR
- The Jenkins pipeline triggers automatically (or click **Build Now** manually if needed)
- Jenkins builds a new image, pushes it to Docker Hub, and updates the K8s manifest in Git
- ArgoCD detects the change and syncs it into the cluster
- Refresh the app in the browser to see the new feature live

Repeat this same flow for subsequent features (`featureB`, etc.), always branching fresh off an updated `main`.

---

## 🧹 Cleanup

Delete the Service and ArgoCD app first to avoid orphaning the Load Balancer:
```bash
kubectl delete application wave-racer -n argocd
```

Then delete the cluster:
```bash
eksctl delete cluster --name wave-racer --region us-east-1
```

Afterward, confirm nothing was left behind:
```bash
aws cloudformation list-stacks --region us-east-1 \
  --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE DELETE_FAILED DELETE_IN_PROGRESS \
  --query "StackSummaries[?contains(StackName, 'wave-racer')].{Name:StackName,Status:StackStatus}"

aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --region us-east-1
aws elb describe-load-balancers --region us-east-1
aws elbv2 describe-load-balancers --region us-east-1
```
