# Steps to Run the MNIST Training Script in Docker

This setup runs `train_mnist.py` inside a Docker container and stores outputs on your host machine.

The training script writes:

- MNIST dataset files to the configured `--data-dir`
- timestamped training outputs to the configured `--output-dir`
- `train.log`
- `checkpoint.pt`

The key requirement is to use Docker bind mounts so those paths map to folders on your host.

Docker Desktop was used to run the application. Ensure you install Docker Desktop or similar (like Miniukube) before running Docker commands. 

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

To create the Image yourself, follow the instructions step by step.

## 1. The project files

Place these files in the same folder:

```text
.
├── Dockerfile
├── requirements.txt
└── train_mnist.py
```

---

## 2. Let's build the Docker image. The Dockerfile is commented for you to understand every line of the file.

From the folder containing the Dockerfile, run:

```bash
docker build -t mnist-trainer:1 .
```

---

## 3. Create host folders for data and results

```bash
mkdir -p data results
```

These folders will live on your host machine.

---

## 4. Run training with outputs saved to the host

### macOS / Linux / WSL

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
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

After the run completes, check your host machine:

```bash
ls -R results
```

You should see a timestamped folder similar to:

```text
results/
└── 20260430_153012/
    ├── checkpoint.pt
    └── train.log
```

---

## 5. Run with custom training arguments

You can override the default command by passing arguments after the image name:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-trainer:1\
  python train_mnist.py \
  --data-dir /app/data \
  --output-dir /app/results \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.001
```

---

## 6. View logs while training

The script prints logs to the terminal and also writes them to `train.log` in the mounted `results` folder.

You can monitor the latest run from the host:

```bash
tail -f results/*/train.log
```

---

## 7. Optional: Run with GPU support

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

## 8. Why the bind mounts matter

Do not rely on container-local paths only. If you run without bind mounts, outputs are written inside the container filesystem and may disappear when the container is removed.

This is the important part:

```bash
-v "$(pwd)/data:/app/data"
-v "$(pwd)/results:/app/results"
```

It maps:

```text
Host folder      Container folder
./data       ->  /app/data
./results    ->  /app/results
```

So the dataset, logs, and model checkpoint remain accessible on the host.
