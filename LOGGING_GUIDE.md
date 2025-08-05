# Date-Based Logging Guide

This guide explains how to implement date-based logging in your Aquarium ML project.

## Overview

Date-based logging creates separate log files for different time periods (daily, hourly, etc.), making it easier to:
- Organize logs by date
- Archive old logs
- Debug issues within specific time periods
- Manage disk space better

## Implemented Solutions

### 1. Current Implementation (Recommended)

**Location**: `main.py` - `setup_date_based_logging()` function

**How it works**:
- Uses Python's `TimedRotatingFileHandler`
- Automatically creates a new log file at midnight each day
- Keeps 30 days of log history
- File naming pattern: `aquarium_ml.log.YYYY-MM-DD`

**Example files created**:
```
logs/
├── aquarium_ml.log              # Current day's log
├── aquarium_ml.log.2025-08-04   # Previous day's log
├── aquarium_ml.log.2025-08-03   # Two days ago
└── ...
```

**Configuration**:
```python
file_handler = TimedRotatingFileHandler(
    filename=log_filename,
    when='midnight',      # Rotate at midnight
    interval=1,          # Every 1 day
    backupCount=30,      # Keep 30 days
    encoding='utf-8'
)
file_handler.suffix = '%Y-%m-%d'  # Date format
```

### 2. Alternative Approaches

**Location**: `src/libs/date_logger.py`

#### Approach 1: Automatic Daily Rotation
- Same as current implementation
- Best for production environments
- Automatic rotation without intervention

#### Approach 2: Date in Filename
- Creates files with date in the name from start
- File pattern: `aquarium_ml_YYYY-MM-DD.log`
- Good for explicit date tracking
- Requires manual cleanup of old files

#### Approach 3: Hourly Rotation
- Creates new log file every hour
- File pattern: `aquarium_ml.log.YYYY-MM-DD_HH`
- Best for high-volume logging
- Keeps 7 days (168 hours) of history

## Testing

Run the test script to see all approaches in action:

```bash
python test_logging.py
```

This will create test log files in `logs/test/` directory.

## Configuration Options

### Rotation Timing
```python
when='midnight'    # Daily at midnight
when='H'          # Every hour
when='M'          # Every minute
when='S'          # Every second
when='W0'         # Weekly on Monday
```

### Backup Count
```python
backupCount=30     # Keep 30 files
backupCount=168    # Keep 168 files (7 days of hourly logs)
backupCount=0      # Keep all files (not recommended)
```

### Date Format
```python
suffix = '%Y-%m-%d'        # 2025-08-05
suffix = '%Y-%m-%d_%H'     # 2025-08-05_14
suffix = '%Y%m%d'          # 20250805
suffix = '%Y-%m-%d_%H-%M'  # 2025-08-05_14-30
```

## Log Format

The current implementation uses an enhanced format:

```python
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
```

**Example output**:
```
2025-08-05 14:30:25,123 - __main__ - INFO - [main.py:125] - ✅ Task scheduler started
2025-08-05 14:30:25,456 - ml_worker - WARNING - [ml_worker.py:67] - ⚠️ Queue is empty
```

## Production Recommendations

1. **Use TimedRotatingFileHandler** (current implementation)
2. **Set appropriate backup count** (30 days for daily logs)
3. **Monitor disk space** regularly
4. **Consider log compression** for archived files
5. **Set up log monitoring** with tools like ELK stack or Grafana

## Monitoring Log Files

### Check current log files:
```bash
ls -la logs/
```

### View today's logs:
```bash
tail -f logs/aquarium_ml.log
```

### View specific date's logs:
```bash
cat logs/aquarium_ml.log.2025-08-04
```

### Find logs by pattern:
```bash
grep "ERROR\|WARNING" logs/aquarium_ml.log*
```

## Integration with Docker

If using Docker, make sure to mount the logs directory:

```yaml
# docker-compose.yml
volumes:
  - ./logs:/app/logs
```

## Cleanup Script

Create a cleanup script for old logs:

```bash
#!/bin/bash
# cleanup_logs.sh
find logs/ -name "*.log.*" -mtime +30 -delete
echo "Cleaned up log files older than 30 days"
```

## Troubleshooting

### Issue: Logs not rotating
- Check file permissions in logs directory
- Verify the application runs continuously past midnight
- Check disk space availability

### Issue: Too many log files
- Reduce `backupCount` value
- Implement cleanup script
- Consider log compression

### Issue: Missing logs
- Check if logs directory exists and is writable
- Verify logging configuration is called before other logging
- Check for competing logging configurations
