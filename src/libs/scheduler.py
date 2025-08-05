import time
import logging
import threading
import schedule
from src.libs.rabbitmq_manager import RabbitMQManager
from src.libs.supabase_manager import SupabaseManager
from src.config.settings import settings
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self):
        self.supabase = SupabaseManager()
        self.rabbitmq = RabbitMQManager()
        self.running = False

    def start(self):
        """Start the task scheduler - REMOVED critical monitoring"""
        try:
            self.rabbitmq.connect()
            self.running = True

            # Schedule tasks
            schedule.every().hour.at(":10").do(self.schedule_model_training)

            schedule.every().hour.at(":15").do(self.schedule_predictions)

            schedule.every().hour.at(":05").do(self.schedule_predict_validation)

            schedule.every().day.at("23:57").do(self.schedule_missing_data)

            if settings.ENVIRONMENT == "development":
                self.force_run_tasks()

            def run_scheduler():
                while self.running:
                    try:
                        schedule.run_pending()
                        time.sleep(30)  # Check every 30 seconds (less frequent)
                    except Exception as e:
                        logger.error(f"Scheduler error: {e}")
                        time.sleep(60)
            
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    def schedule_predict_validation(self):
        """Schedule prediction validation every 30 minutes"""
        try:
            aquariums = self.supabase.get_active_aquariums()

            target_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'validate_predictions',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'priority': 50,
                    'target_time': target_time.isoformat(),
                }
                self.rabbitmq.safe_publish('validate_predictions', task_data)
            
            logger.info(f"Scheduled prediction validation for {len(aquariums)} aquariums")
            
        except Exception as e:
            logger.error(f"Error scheduling prediction validation: {e}")

    def schedule_anomaly_detection(self):
        """Schedule anomaly detection every 30 minutes"""
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'anomaly_detection',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'priority': 200
                }
                self.rabbitmq.safe_publish('anomaly_detection', task_data)
            
            logger.info(f"Scheduled anomaly detection for {len(aquariums)} aquariums")
            
        except Exception as e:
            logger.error(f"Error scheduling anomaly detection: {e}")

    def schedule_predictions(self):
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'predictions',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'interval_minutes': 30,
                    'priority': 150,
                    'date_time_now': datetime.now(timezone.utc).isoformat()
                }
                self.rabbitmq.safe_publish('predictions', task_data)
            
            logger.info(f"Scheduled predictions for {len(aquariums)} aquariums at {datetime.now(timezone.utc).isoformat()}")
            
        except Exception as e:
            logger.error(f"Error scheduling predictions: {e}")

    def schedule_model_training(self):
        """Schedule model training for all active aquariums"""
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'model_training',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'parameters': ['ph', 'do', 'water_temperature'],
                    'priority': 100  # Lowest priority
                }
                self.rabbitmq.safe_publish('model_training', task_data)
            
            logger.info(f"Scheduled model training for {len(aquariums)} aquariums")
            
        except Exception as e:
            logger.error(f"Error scheduling model training: {e}")

    def schedule_missing_data(self):
        """Schedule missing data task for all active aquariums"""
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            date_time_end = datetime.now(timezone.utc)
            date_time_start = date_time_end.replace(hour=0, minute=0, second=0, microsecond=0)

            for aquarium in aquariums:
                task_data = {
                    'task_type': 'missing_data',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'priority': 250,  # Highest priority,
                    'date_time_start': date_time_start.isoformat(),
                    'date_time_end': date_time_end.isoformat()
                }
                self.rabbitmq.safe_publish('missing_data', task_data)
            
            logger.info(f"Scheduled missing data task for {len(aquariums)} aquariums")
            
        except Exception as e:
            logger.error(f"Error scheduling missing data task: {e}")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        
        # Clear all scheduled jobs
        schedule.clear()
        
        # Close RabbitMQ connection
        self.rabbitmq.close()
        
        logger.info("Task scheduler stopped")

    def health_check(self) -> dict:
        """Check scheduler health"""
        try:
            return {
                'running': self.running,
                'scheduled_jobs': len(schedule.jobs),
                'rabbitmq_connected': self.rabbitmq.connection and not self.rabbitmq.connection.is_closed,
                'next_run': schedule.next_run().isoformat() if schedule.next_run() else None, # type: ignore
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in scheduler health check: {e}")
            return {
                'running': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
    def force_run_tasks(self):
        """Force run tasks immediately (for testing)"""
        try:
            logger.info("Force running scheduled tasks...")
            
            self.schedule_anomaly_detection()
            time.sleep(1)
            
            self.schedule_predictions()
            time.sleep(1)
            
            self.schedule_model_training()
            time.sleep(1)

            self.schedule_predict_validation()
            
            logger.info("All tasks force-scheduled successfully")
            
        except Exception as e:
            logger.error(f"Error force running tasks: {e}")