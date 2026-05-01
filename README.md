# Steps to Run the MNIST Training Script in Docker

## Introduction

This setup runs `train_mnist.py` inside a Docker container and stores outputs on your host machine.

The training script writes:

- MNIST dataset files to the configured `--data-dir`
- timestamped training outputs to the configured `--output-dir`
    - `train.log`
    - `checkpoint.pt`

The key requirement is to use Docker bind mounts so those paths map to folders on your host. As per the instructions, you don't have to map the data directory to the laptop but if you need it, include it.

Docker Desktop was used to run the application. Ensure you install Docker Desktop or similar (like Miniukube) before running Docker commands. Also ensure you have git client installed.


---

# How to build the image

Follow the instructions step by step.

## 1. Clone the Repository 


```text
git clone https://github.com/igbedo/ml-infra-workflow-assessment-solution.git
```

---

## 2. Build the Docker image. 

The Dockerfile is commented for you to understand every line of the file.

From the folder containing the Dockerfile, run:
```bash
cd ml-infra-workflow-assessment-solution
```
```bash
docker build -t mnist-trainer:1 .
```

---

# How to run the container for training with outputs saved to the host

### macOS / Linux / WSL

With data directory:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-trainer:1
```

No data directory:
```bash
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  mnist-trainer:1
```

### Windows PowerShell

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/results:/app/results" `
  mnist-trainer:1
```

# View the Output of the Training run in the results folder

Note that every run creates an additional timestemp folder in the results folder so you may see several folders depending on the number of runs.

After the run completes, check your host machine:

```bash
ls -R results
```

You should see a timestamped folder similar to:


```text
results/
└── 20260430_233307/
    ├── checkpoint.pt
    └── train.log
```

---


## Optional: View logs while training

The script prints logs to the terminal and also writes them to `train.log` in the mounted `results` folder.

You can monitor the latest run from the host:

```bash
tail -f results/*/train.log
```

---

## Optional: Run with GPU support

The provided Dockerfile works for CPU training.

For GPU training, you need:

1. NVIDIA GPU drivers installed on the host
2. NVIDIA Container Toolkit installed
3. A CUDA-compatible PyTorch image or CUDA-enabled PyTorch install

Example GPU run command, after preparing a GPU-compatible image:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-trainer:2
```

Inside the script, PyTorch automatically selects CUDA when available.

---


## Optional: You can use my image pushed to Docker Hub

I already pushed the image to DockerHub so you can run with just 2 steps:

Step 1:
```bash
mkdir -p data results
```

Step 2:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  igbedo/mnist-trainer:1
```

---