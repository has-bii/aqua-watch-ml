import threading
import time
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from src.libs.scheduler import TaskScheduler
from src.libs.ml_worker import MLWorker
from src.config.settings import settings


def setup_date_based_logging():
    """
    Setup logging with date-based file rotation.
    
    This creates log files that rotate daily at midnight.
    Files will be named: aquarium_ml.log.2025-08-05, aquarium_ml.log.2025-08-06, etc.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(settings.LOG_PATH, exist_ok=True)
    
    # Create formatter with more detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create date-based file handler
    log_filename = os.path.join(settings.LOG_PATH, 'aquarium_ml.log')
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',           # Rotate at midnight
        interval=1,               # Every 1 day
        backupCount=30,           # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = '%Y-%m-%d'  # Date format: YYYY-MM-DD
    
    # Create console handler with slightly simpler format
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log the logging setup
    logging.info(f"📝 Date-based logging initialized. Log files will be created in: {settings.LOG_PATH}")
    logging.info(f"📅 Current log file: {log_filename}")
    logging.info(f"🔄 Log rotation: Daily at midnight, keeping 30 days of history")
    
    return logger

# Setup date-based logging
setup_date_based_logging()

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(settings.LOG_PATH, exist_ok=True)
    os.makedirs(settings.DATA_CACHE_PATH, exist_ok=True)

def start_workers():
    """Start ML workers"""
    workers = [
        ('anomaly_detection', 1),
        ('predictions', 1),
        ('validate_predictions', 1),
        ('model_training', 1),
        ('missing_data', 1)
    ]
    
    worker_threads = []
    
    for queue_name, worker_count in workers:
        for i in range(worker_count):
            worker_id = f"{queue_name}_worker_{i}"
            worker = MLWorker(worker_id)
            
            def start_worker(worker_instance, queue):
                try:
                    worker_instance.start_consuming(queue)
                except Exception as e:
                    logger.error(f"Worker {worker_instance.worker_id} failed: {e}")
            
            thread = threading.Thread(
                target=start_worker,
                args=(worker, queue_name),
                daemon=True,
                name=worker_id
            )
            thread.start()
            worker_threads.append(thread)
            logger.info(f"Started worker: {worker_id}")
    
    return worker_threads

def check_dependencies():
    """Check if all required dependencies are available"""
    checks = {
        # "Environment file": os.path.exists('.env'),
        "Supabase URL": settings.SUPABASE_URL is not None,
        "Supabase Key": settings.SUPABASE_SERVICE_KEY is not None,
        "RabbitMQ URL": settings.RABBITMQ_URL is not None,
    }
    
    logger.info("Dependency Check:")
    all_good = True
    for check, status in checks.items():
        status_text = "✅ OK" if status else "❌ MISSING"
        logger.info(f"  {check}: {status_text}")
        if not status:
            all_good = False
    
    if not all_good:
        logger.error("Some dependencies are missing. Please check your configuration.")
        return False
    
    return True

def print_startup_banner():
    """Print startup banner"""
    banner = """
    🐠 ====================================================
    🐠    AQUARIUM ML MONITORING SYSTEM
    🐠 ====================================================
    """
    print(banner)
    print(f"📅 Started at: {datetime.now()}")
    print(f"🗂️  Model Path: {settings.MODEL_SAVE_PATH}")
    print(f"📋 Log Path: {settings.LOG_PATH}")
    print(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")

def main():
    """Main application entry point"""
    print_startup_banner()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting...")
        return
    
    # Create directories
    create_directories()
    logger.info("📁 Created necessary directories")
    
    # Start task scheduler
    try:
        scheduler = TaskScheduler()
        scheduler.start()
        logger.info("✅ Task scheduler started")
    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {e}")
        return
    
    # Start ML workers
    try:
        worker_threads = start_workers()
        logger.info(f"✅ Started {len(worker_threads)} ML workers")
    except Exception as e:
        logger.error(f"❌ Failed to start workers: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 AQUARIUM ML SYSTEM IS FULLY OPERATIONAL!")

    print("\n💡 Tip: Use Ctrl+C to stop the system gracefully")
    print("\n🔍 System is now monitoring your aquariums 24/7!")
    
    # Monitor system health
    def health_monitor():
        check_count = 0
        while True:
            try:
                check_count += 1
                
                # Check worker threads every minute
                alive_workers = sum(1 for t in worker_threads if t.is_alive())
                if alive_workers < len(worker_threads):
                    logger.warning(f"Only {alive_workers}/{len(worker_threads)} workers alive")
                
                # Every 10 minutes, log system status
                if check_count % 10 == 0:
                    logger.info(f"System health check: {alive_workers}/{len(worker_threads)} workers alive")
                    
                    # Check scheduler health
                    scheduler_health = scheduler.health_check()
                    if not scheduler_health['running']:
                        logger.error("Scheduler is not running!")
                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(60)
    
    # Start health monitor
    health_thread = threading.Thread(target=health_monitor, daemon=True)
    health_thread.start()
    
    # Signal handler for graceful shutdown
    import signal
    
    def signal_handler(sig, frame):
        print(f"\n🛑 Received signal {sig}. Shutting down gracefully...")
        logger.info(f"Received signal {sig}. Starting graceful shutdown...")
        
        try:
            # Stop scheduler first
            scheduler.stop()
            logger.info("✅ Scheduler stopped")
            
            # Wait a moment for workers to finish current tasks
            logger.info("⏳ Waiting for workers to finish current tasks...")
            time.sleep(5)
            
            logger.info("👋 Shutdown complete. Goodbye!")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()