# Docker Deployment Guide 🐳

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU support)

## Installation NVIDIA Container Toolkit

Jika belum terinstall, jalankan:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

## Quick Start

### 1. Build dan Run dengan Docker Compose

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### 2. Build dan Run dengan Docker (tanpa compose)

```bash
# Build image
docker build -t face-swap-api .

# Run container
docker run -d \
  --name face-swap-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/models:/app/models \
  face-swap-api
```

## Akses API

Setelah container berjalan, API dapat diakses di:

- **API URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Volumes

Container menggunakan 3 volumes:

- `./outputs` - Untuk menyimpan hasil face swap
- `./uploads` - Untuk temporary upload files
- `./models` - Untuk menyimpan model InsightFace (akan auto-download saat pertama kali run)

## Monitoring

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f face-swap-api

# Check GPU usage inside container
docker exec -it face-swap-api nvidia-smi

# Enter container shell
docker exec -it face-swap-api bash
```

## Environment Variables

Anda dapat menambahkan environment variables di `docker-compose.yml`:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - LOG_LEVEL=info
  - MAX_FILE_SIZE=10485760  # 10MB
```

## Troubleshooting

### GPU tidak terdeteksi

```bash
# Test GPU di container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Port 8000 sudah digunakan

Edit `docker-compose.yml`, ubah mapping port:
```yaml
ports:
  - "8001:8000"  # Gunakan port 8001 di host
```

### Rebuild container setelah update code

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Production Tips

1. **Resource Limits**: Tambahkan resource limits di docker-compose.yml
2. **Restart Policy**: Sudah dikonfigurasi dengan `restart: unless-stopped`
3. **Health Check**: Health check sudah dikonfigurasi untuk monitoring
4. **Logging**: Gunakan Docker logging driver untuk centralized logs

## Testing API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test swap faces (contoh)
curl -X POST "http://localhost:8000/swap-faces" \
  -F "target=@target.jpg" \
  -F "source1=@source1.jpg" \
  -F "mode=sequential"
```
