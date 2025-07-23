#!/bin/bash

# Grafana Dashboard Setup Script for Aqua Watch ML
# This script helps automate the Grafana dashboard setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASS="admin123"
PROMETHEUS_URL="http://prometheus:9090"

# Function to check if Grafana is running
check_grafana_status() {
    print_header "Checking Grafana Status"
    
    if curl -s -f "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
        print_status "Grafana is running at $GRAFANA_URL"
        return 0
    else
        print_error "Grafana is not accessible at $GRAFANA_URL"
        print_warning "Make sure you've started the monitoring stack:"
        print_warning "  docker-compose --profile monitoring up -d"
        return 1
    fi
}

# Function to add Prometheus data source
add_prometheus_datasource() {
    print_header "Adding Prometheus Data Source"
    
    # Check if data source already exists
    if curl -s -u "$GRAFANA_USER:$GRAFANA_PASS" "$GRAFANA_URL/api/datasources/name/Prometheus" > /dev/null 2>&1; then
        print_warning "Prometheus data source already exists"
        return 0
    fi
    
    # Create Prometheus data source
    curl -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASS" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "'$PROMETHEUS_URL'",
            "access": "proxy",
            "isDefault": true
        }' \
        "$GRAFANA_URL/api/datasources"
    
    if [ $? -eq 0 ]; then
        print_status "Prometheus data source added successfully"
    else
        print_error "Failed to add Prometheus data source"
        return 1
    fi
}

# Function to import dashboard
import_dashboard() {
    local dashboard_file="$1"
    local dashboard_name="$2"
    
    print_status "Importing dashboard: $dashboard_name"
    
    if [ ! -f "$dashboard_file" ]; then
        print_error "Dashboard file not found: $dashboard_file"
        return 1
    fi
    
    # Prepare dashboard JSON
    local dashboard_json=$(cat "$dashboard_file")
    local import_payload="{\"dashboard\": $dashboard_json, \"overwrite\": true}"
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -u "$GRAFANA_USER:$GRAFANA_PASS" \
        -d "$import_payload" \
        "$GRAFANA_URL/api/dashboards/db" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_status "Dashboard '$dashboard_name' imported successfully"
    else
        print_error "Failed to import dashboard: $dashboard_name"
        return 1
    fi
}

# Function to import all dashboards
import_all_dashboards() {
    print_header "Importing Dashboards"
    
    local dashboard_dir="./grafana/dashboards"
    
    if [ ! -d "$dashboard_dir" ]; then
        print_error "Dashboard directory not found: $dashboard_dir"
        return 1
    fi
    
    # Import each dashboard
    import_dashboard "$dashboard_dir/aqua-watch-overview.json" "Aqua Watch System Overview"
    import_dashboard "$dashboard_dir/rabbitmq-dashboard.json" "RabbitMQ Monitoring"
    import_dashboard "$dashboard_dir/ml-pipeline-dashboard.json" "ML Pipeline Performance"
}

# Function to enable RabbitMQ Prometheus plugin
enable_rabbitmq_metrics() {
    print_header "Enabling RabbitMQ Prometheus Plugin"
    
    if docker exec aqua-watch-rabbitmq rabbitmq-plugins list | grep -q "rabbitmq_prometheus.*E"; then
        print_status "RabbitMQ Prometheus plugin is already enabled"
    else
        print_status "Enabling RabbitMQ Prometheus plugin..."
        docker exec aqua-watch-rabbitmq rabbitmq-plugins enable rabbitmq_prometheus
        
        if [ $? -eq 0 ]; then
            print_status "RabbitMQ Prometheus plugin enabled successfully"
            print_warning "Waiting for RabbitMQ to restart..."
            sleep 10
        else
            print_error "Failed to enable RabbitMQ Prometheus plugin"
            return 1
        fi
    fi
}

# Function to show access information
show_access_info() {
    print_header "Access Information"
    echo -e "${BLUE}🌐 Grafana Dashboard:${NC} $GRAFANA_URL"
    echo -e "${BLUE}👤 Username:${NC} $GRAFANA_USER"
    echo -e "${BLUE}🔑 Password:${NC} $GRAFANA_PASS"
    echo ""
    echo -e "${BLUE}📊 Other Services:${NC}"
    echo -e "   Prometheus: http://localhost:9090"
    echo -e "   RabbitMQ Management: http://localhost:15672 (aquawatch/aquawatch123)"
    echo ""
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo -e "Your Grafana dashboards are ready for monitoring your Aqua Watch ML system."
}

# Function to show help
show_help() {
    echo "Grafana Setup Script for Aqua Watch ML"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup          Complete setup (data source + dashboards + RabbitMQ)"
    echo "  datasource     Add Prometheus data source only"
    echo "  dashboards     Import dashboards only"
    echo "  rabbitmq       Enable RabbitMQ metrics only"
    echo "  status         Check Grafana status"
    echo "  info           Show access information"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup       # Complete automated setup"
    echo "  $0 status      # Check if Grafana is running"
}

# Main script logic
case "${1:-setup}" in
    "setup")
        print_header "Aqua Watch ML - Grafana Setup"
        check_grafana_status || exit 1
        enable_rabbitmq_metrics
        add_prometheus_datasource
        import_all_dashboards
        show_access_info
        ;;
    "datasource")
        check_grafana_status || exit 1
        add_prometheus_datasource
        ;;
    "dashboards")
        check_grafana_status || exit 1
        import_all_dashboards
        ;;
    "rabbitmq")
        enable_rabbitmq_metrics
        ;;
    "status")
        check_grafana_status
        ;;
    "info")
        show_access_info
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
