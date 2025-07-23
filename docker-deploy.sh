#!/bin/bash

# Aqua Watch ML Docker Deployment Script
# This script helps with common Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if .env file exists
check_env_file() {
    if [[ ! -f .env ]]; then
        print_warning ".env file not found!"
        if [[ -f .env.example ]]; then
            print_status "Copying .env.example to .env"
            cp .env.example .env
            print_warning "Please edit .env file with your actual configuration!"
            return 1
        else
            print_error ".env.example not found. Please create .env file manually."
            return 1
        fi
    fi
    return 0
}

# Function to start services
start_services() {
    local compose_file=${1:-docker-compose.yml}
    local profile=${2:-""}
    
    print_header "Starting Aqua Watch ML Services"
    
    if ! check_env_file; then
        print_error "Please configure .env file before starting services"
        exit 1
    fi
    
    local cmd="docker-compose -f $compose_file"
    
    if [[ -n "$profile" ]]; then
        cmd="$cmd --profile $profile"
    fi
    
    print_status "Starting services with: $cmd up -d"
    $cmd up -d
    
    print_status "Services started successfully!"
    print_status "Use '$0 status' to check service health"
}

# Function to stop services
stop_services() {
    local compose_file=${1:-docker-compose.yml}
    
    print_header "Stopping Aqua Watch ML Services"
    docker-compose -f $compose_file down
    print_status "Services stopped successfully!"
}

# Function to show service status
show_status() {
    print_header "Service Status"
    docker-compose ps
    
    print_header "Resource Usage"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
}

# Function to show logs
show_logs() {
    local service=${1:-aqua-watch-ml}
    local lines=${2:-100}
    
    print_header "Showing logs for $service (last $lines lines)"
    docker-compose logs --tail=$lines -f $service
}

# Function to backup data
backup_data() {
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_header "Backing up data to $backup_dir"
    
    # Backup models
    if docker volume ls | grep -q "aqua-watch-ml_model_data"; then
        print_status "Backing up model data..."
        docker run --rm \
            -v aqua-watch-ml_model_data:/data \
            -v "$(pwd)/$backup_dir":/backup \
            alpine tar czf /backup/models.tar.gz -C /data .
    fi
    
    # Backup logs
    if docker volume ls | grep -q "aqua-watch-ml_log_data"; then
        print_status "Backing up log data..."
        docker run --rm \
            -v aqua-watch-ml_log_data:/data \
            -v "$(pwd)/$backup_dir":/backup \
            alpine tar czf /backup/logs.tar.gz -C /data .
    fi
    
    print_status "Backup completed: $backup_dir"
}

# Function to update services
update_services() {
    print_header "Updating Aqua Watch ML Services"
    
    print_status "Pulling latest images..."
    docker-compose pull
    
    print_status "Rebuilding application image..."
    docker-compose build --no-cache aqua-watch-ml
    
    print_status "Restarting services..."
    docker-compose up -d
    
    print_status "Cleaning up old images..."
    docker image prune -f
    
    print_status "Update completed!"
}

# Function to clean up
cleanup() {
    print_header "Cleaning up Docker resources"
    
    print_warning "This will remove stopped containers, unused networks, and dangling images"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker system prune -f
        print_status "Cleanup completed!"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Aqua Watch ML Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [compose-file]     Start services (default: docker-compose.yml)"
    echo "  stop [compose-file]      Stop services"
    echo "  restart [compose-file]   Restart services"
    echo "  status                   Show service status and resource usage"
    echo "  logs [service] [lines]   Show logs (default: aqua-watch-ml, 100 lines)"
    echo "  backup                   Backup data volumes"
    echo "  update                   Update and restart services"
    echo "  cleanup                  Clean up Docker resources"
    echo "  dev                      Start development environment"
    echo "  prod                     Start production environment"
    echo "  monitoring               Start with monitoring stack"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                 # Start development environment"
    echo "  $0 prod                  # Start production environment"
    echo "  $0 logs aqua-watch-ml 50 # Show last 50 logs for main service"
    echo "  $0 monitoring            # Start with Prometheus and Grafana"
}

# Main script logic
case "${1:-help}" in
    "start")
        start_services "${2:-docker-compose.yml}"
        ;;
    "stop")
        stop_services "${2:-docker-compose.yml}"
        ;;
    "restart")
        stop_services "${2:-docker-compose.yml}"
        start_services "${2:-docker-compose.yml}"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2" "$3"
        ;;
    "backup")
        backup_data
        ;;
    "update")
        update_services
        ;;
    "cleanup")
        cleanup
        ;;
    "dev")
        start_services "docker-compose.yml"
        ;;
    "prod")
        start_services "docker-compose.prod.yml"
        ;;
    "monitoring")
        start_services "docker-compose.yml" "monitoring"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
