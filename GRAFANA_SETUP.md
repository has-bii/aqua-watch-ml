# Grafana Dashboard Setup Guide for Aqua Watch ML

This guide will walk you through setting up comprehensive monitoring dashboards for your Aqua Watch ML system.

## 🚀 Quick Access

- **Grafana Dashboard**: http://localhost:3000
- **Default Login**: admin / admin123
- **Prometheus**: http://localhost:9090
- **RabbitMQ Management**: http://localhost:15672 (aquawatch / aquawatch123)

## 📊 Step-by-Step Setup

### Step 1: Access Grafana

1. Open your browser and go to: http://localhost:3000
2. Login with:
   - Username: `admin`
   - Password: `admin123` (or change in .env file with GRAFANA_PASSWORD)

### Step 2: Add Prometheus Data Source

1. Click on the gear icon (⚙️) in the left sidebar → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Configure the connection:
   - **Name**: `Prometheus`
   - **URL**: `http://prometheus:9090`
   - **Access**: `Server (default)`
5. Click **Save & Test** - you should see "Data source is working"

### Step 3: Import Pre-built Dashboards

I've created three custom dashboards for you. Import them using these methods:

#### Method 1: Manual Import (Recommended)

1. Go to **+** (plus icon) → **Import** in the left sidebar
2. Copy and paste the JSON content from each dashboard file:
   - `grafana/dashboards/aqua-watch-overview.json` - System Overview
   - `grafana/dashboards/rabbitmq-dashboard.json` - RabbitMQ Monitoring
   - `grafana/dashboards/ml-pipeline-dashboard.json` - ML Pipeline Performance

#### Method 2: File Upload

1. Go to **+** → **Import**
2. Click **Upload JSON file**
3. Select each dashboard file from the `grafana/dashboards/` directory

### Step 4: Configure Dashboard Settings

For each imported dashboard:

1. **Data Source**: Select "Prometheus" as the data source
2. **Refresh Rate**: Set to 30s or 1m for real-time monitoring
3. **Time Range**: Default is "Last 1 hour" - adjust as needed

## 📈 Dashboard Overview

### 1. Aqua Watch System Overview
- **System Status**: Overall health of your ML system
- **RabbitMQ Queue Status**: Message queue monitoring
- **Container Resources**: CPU and memory usage
- **ML Worker Status**: Active consumers and workers
- **Application Logs**: Real-time log viewing

### 2. RabbitMQ Monitoring
- **Queue Messages**: Ready, unacknowledged, and total messages
- **Message Rates**: Publishing and delivery rates
- **Consumer Count**: Number of active workers per queue
- **Connection/Channel Counts**: RabbitMQ connection health
- **Memory Usage**: RabbitMQ node memory consumption

### 3. ML Pipeline Performance
- **Job Queues**: Pending jobs for training, predictions, and anomaly detection
- **Processing Rates**: Jobs processed per second
- **Worker Health**: Status of ML workers

## 🔧 Advanced Configuration

### Enable RabbitMQ Metrics (Required for full monitoring)

1. **Enable RabbitMQ Prometheus Plugin**:
   ```bash
   # Connect to RabbitMQ container
   docker exec -it aqua-watch-rabbitmq bash
   
   # Enable the prometheus plugin
   rabbitmq-plugins enable rabbitmq_prometheus
   ```

2. **Update Prometheus Configuration**:
   The prometheus.yml is already configured to scrape RabbitMQ metrics on port 15692.

### Add Custom Metrics to Your Application

To get more detailed ML metrics, add Prometheus metrics to your Python application:

```python
# Add to your main.py or create a metrics module
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
model_training_counter = Counter('ml_models_trained_total', 'Total models trained')
prediction_duration = Histogram('ml_prediction_duration_seconds', 'Time spent on predictions')
anomaly_detection_gauge = Gauge('ml_anomalies_detected', 'Number of anomalies detected')

# Start metrics server (add to main.py)
start_http_server(8000)  # Metrics available at http://localhost:8000/metrics
```

### Create Custom Panels

1. **Add New Panel**: Click **Add panel** in any dashboard
2. **Query Builder**: Use PromQL queries like:
   - `rate(rabbitmq_queue_messages_published_total[5m])` - Message publish rate
   - `rabbitmq_queue_messages_ready` - Ready messages in queue
   - `container_memory_usage_bytes{name="aqua-watch-ml-app"}` - Container memory

3. **Visualization Options**:
   - **Graph**: Time series data
   - **Stat**: Single value metrics
   - **Table**: Tabular data
   - **Logs**: Log aggregation

## 🚨 Setting Up Alerts

### 1. Create Alert Rules

1. Go to **Alerting** → **Alert Rules**
2. Click **New Rule**
3. Example alert for high queue size:
   ```
   Query: rabbitmq_queue_messages_ready > 100
   Condition: IS ABOVE 100
   Evaluation: every 1m for 5m
   ```

### 2. Notification Channels

1. Go to **Alerting** → **Notification channels**
2. Add channels for:
   - **Email**: SMTP configuration
   - **Slack**: Webhook URL
   - **Discord**: Webhook URL

## 📊 Key Metrics to Monitor

### System Health
- Container CPU/Memory usage
- Disk space utilization
- Network I/O

### RabbitMQ
- Queue depth (messages ready)
- Consumer count per queue
- Message publish/consume rates
- Connection count

### ML Pipeline
- Job processing time
- Model training frequency
- Prediction accuracy trends
- Anomaly detection rates

## 🛠️ Troubleshooting

### Common Issues

1. **No Data in Dashboards**
   - Check Prometheus data source connection
   - Verify services are running: `docker-compose ps`
   - Check Prometheus targets: http://localhost:9090/targets

2. **RabbitMQ Metrics Missing**
   - Enable prometheus plugin in RabbitMQ
   - Check port 15692 is accessible

3. **Application Metrics Missing**
   - Add Prometheus client to your Python app
   - Expose metrics endpoint on port 8000

### Useful Commands

```bash
# Check service status
docker-compose ps

# View Grafana logs
docker-compose logs grafana

# View Prometheus logs
docker-compose logs prometheus

# Access RabbitMQ management
open http://localhost:15672

# Check Prometheus targets
open http://localhost:9090/targets
```

## 🎨 Dashboard Customization

### Themes and Styling
1. Go to **Settings** → **Preferences**
2. Choose **Dark** or **Light** theme
3. Set default dashboard and timezone

### Panel Customization
- **Colors**: Customize color schemes for better visibility
- **Units**: Set appropriate units (bytes, percentage, etc.)
- **Thresholds**: Set warning/critical thresholds with color coding
- **Legends**: Customize legend display and positioning

## 📱 Mobile Access

Grafana dashboards are mobile-responsive. Access via:
- **Mobile Browser**: http://your-server-ip:3000
- **Grafana Mobile App**: Available on iOS/Android

## 🔐 Security Best Practices

1. **Change Default Password**: Update admin password
2. **User Management**: Create specific users for different roles
3. **HTTPS**: Configure SSL/TLS for production
4. **API Keys**: Use API keys for external integrations

## 📈 Performance Optimization

1. **Query Optimization**: Use efficient PromQL queries
2. **Refresh Intervals**: Don't set refresh rates too low
3. **Time Ranges**: Limit historical data ranges for better performance
4. **Panel Limits**: Limit number of series in graphs

This setup provides comprehensive monitoring for your Aqua Watch ML system. The dashboards will help you track system health, identify bottlenecks, and ensure optimal performance of your ML pipeline.
