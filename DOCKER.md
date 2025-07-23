# Docker Deployment Guide for Aqua Watch ML

This guide explains how to deploy the Aqua Watch ML system using Docker and Docker Compose.

## Quick Start

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd aqua-watch-ml
   ```

2. **Copy the environment template and configure your settings**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration values
   ```

3. **Start the services**
   ```bash
   # For development with all services
   docker-compose up -d
   
   # For production (external services)
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Architecture

The Docker setup includes the following services:

### Core Services
- **aqua-watch-ml**: Main ML application container
- **rabbitmq**: Message broker for task queuing
- **redis**: (Optional) Caching layer for improved performance

### Monitoring Services (Optional)
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboard

## Environment Configuration

### Required Environment Variables

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key-here

# RabbitMQ Configuration
RABBITMQ_URL=amqp://username:password@host:5672/
RABBITMQ_PASSWORD=your-password

# Optional: Redis
REDIS_PASSWORD=your-redis-password
```

## Deployment Options

### 1. Development Deployment

Use the main `docker-compose.yml` for development:

```bash
# Start all services including RabbitMQ
docker-compose up -d

# View logs
docker-compose logs -f aqua-watch-ml

# Stop services
docker-compose down
```

**Services included:**
- Main ML application
- RabbitMQ (with management UI at http://localhost:15672)
- Redis (optional)

### 2. Production Deployment

Use `docker-compose.prod.yml` for production with external services:

```bash
# Start only the ML application (assumes external RabbitMQ/Redis)
docker-compose -f docker-compose.prod.yml up -d

# With custom environment file
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

**Features:**
- Resource limits and reservations
- Enhanced security options
- Log rotation
- Health checks
- Restart policies

### 3. Monitoring Stack

Enable monitoring services:

```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d
```

**Access points:**
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090

## Docker Commands

### Building and Running

```bash
# Build the application image
docker build -t aqua-watch-ml .

# Run standalone container
docker run -d \
  --name aqua-watch-ml \
  --env-file .env \
  -v $(pwd)/models:/app/models/saved_models \
  -v $(pwd)/logs:/app/logs \
  aqua-watch-ml
```

### Management Commands

```bash
# View container logs
docker-compose logs -f aqua-watch-ml

# Execute commands in container
docker-compose exec aqua-watch-ml bash

# Restart services
docker-compose restart aqua-watch-ml

# Update and restart
docker-compose pull && docker-compose up -d

# Clean up
docker-compose down -v  # Warning: removes volumes
```

### Scaling

```bash
# Scale ML workers (if designed for horizontal scaling)
docker-compose up -d --scale aqua-watch-ml=3
```

## Volume Management

The setup uses several volumes for data persistence:

- `model_data`: ML model files
- `log_data`: Application logs
- `cache_data`: Temporary cache data
- `rabbitmq_data`: RabbitMQ message persistence
- `redis_data`: Redis data persistence

### Backup Volumes

```bash
# Backup models
docker run --rm -v aqua-watch-ml_model_data:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore models
docker run --rm -v aqua-watch-ml_model_data:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /data
```

## Health Checks

The application includes health checks:

```bash
# Check health status
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' aqua-watch-ml-app
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   # Check logs
   docker-compose logs aqua-watch-ml
   
   # Check environment variables
   docker-compose config
   ```

2. **Permission issues**
   ```bash
   # Fix ownership (Linux/Mac)
   sudo chown -R $USER:$USER ./models ./logs ./data
   ```

3. **RabbitMQ connection issues**
   ```bash
   # Check RabbitMQ status
   docker-compose logs rabbitmq
   
   # Access RabbitMQ management
   # http://localhost:15672 (aquawatch/aquawatch123)
   ```

4. **Memory issues**
   ```bash
   # Increase memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

### Performance Tuning

1. **Resource Allocation**
   - Adjust CPU and memory limits based on your hardware
   - Monitor resource usage with `docker stats`

2. **Storage Optimization**
   - Use SSD storage for model and cache volumes
   - Consider using tmpfs for temporary data

3. **Network Optimization**
   - Use bridge networks for service communication
   - Consider host networking for high-throughput scenarios

## Security Best Practices

1. **Environment Variables**
   - Never commit .env files to version control
   - Use Docker secrets in production
   - Rotate credentials regularly

2. **Container Security**
   - Run containers as non-root user
   - Use security options: `no-new-privileges`
   - Keep base images updated

3. **Network Security**
   - Use custom bridge networks
   - Limit exposed ports
   - Consider reverse proxy for HTTPS

## Monitoring and Maintenance

### Log Management

```bash
# View recent logs
docker-compose logs --tail=100 -f aqua-watch-ml

# Log rotation is configured in production compose
# Max 100MB per file, 5 files retained
```

### System Monitoring

```bash
# Resource usage
docker stats

# System health
docker system df
docker system prune  # Clean unused resources
```

### Updates

```bash
# Update images
docker-compose pull

# Restart with new images
docker-compose up -d

# Clean old images
docker image prune
```

## Production Checklist

- [ ] Environment variables configured
- [ ] External services (RabbitMQ, Redis) set up
- [ ] SSL/TLS certificates configured
- [ ] Monitoring and alerting enabled
- [ ] Backup strategy implemented
- [ ] Log aggregation configured
- [ ] Resource limits set appropriately
- [ ] Health checks verified
- [ ] Security settings applied

## Support

For issues and questions:
1. Check the application logs: `docker-compose logs aqua-watch-ml`
2. Verify environment configuration
3. Check service health: `docker-compose ps`
4. Review this documentation for troubleshooting steps
