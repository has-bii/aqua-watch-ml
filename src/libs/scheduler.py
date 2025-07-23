import time
import logging
import threading
import schedule
from src.libs.rabbitmq_manager import RabbitMQManager
from src.libs.supabase_manager import SupabaseManager
from src.config.settings import settings
from datetime import datetime

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
            
            schedule.every(30).minutes.do(self.schedule_anomaly_detection)        # Every 30 minutes
            schedule.every(30).minutes.do(self.schedule_predictions)              # Every 30 minutes
            schedule.every(30).minutes.do(self.schedule_model_training)     # Daily training
            # schedule.every().day.at("00:00").do(self.schedule_model_training)     # Daily training
            
            self.schedule_anomaly_detection()  # Initial run
            self.schedule_predictions()         # Initial run
            self.schedule_model_training()      # Initial run

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


    def schedule_anomaly_detection(self):
        """Schedule anomaly detection every 30 minutes"""
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'anomaly_detection',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'interval_minutes': 30,  # Fixed to 30 minutes
                    'priority': 200
                }
                self.rabbitmq.publish_task('anomaly_detection', task_data)
            
            logger.info(f"Scheduled anomaly detection for {len(aquariums)} aquariums (30-min interval)")
            
        except Exception as e:
            logger.error(f"Error scheduling anomaly detection: {e}")

    def schedule_predictions(self):
        """Schedule predictions every 30 minutes"""
        try:
            aquariums = self.supabase.get_active_aquariums()
            
            for aquarium in aquariums:
                task_data = {
                    'task_type': 'predictions',
                    'aquarium_id': aquarium['id'],
                    'user_id': aquarium['user_id'],
                    'parameters': ['ph', 'do', 'water_temperature'],
                    'interval_minutes': 30,  # Fixed to 30 minutes
                    'priority': 150
                }
                self.rabbitmq.publish_task('predictions', task_data)
            
            logger.info(f"Scheduled predictions for {len(aquariums)} aquariums (30-min interval)")
            
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
                self.rabbitmq.publish_task('model_training', task_data)
            
            logger.info(f"Scheduled model training for {len(aquariums)} aquariums")
            
        except Exception as e:
            logger.error(f"Error scheduling model training: {e}")

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
        """Force run tasks immediately (for testing) - REMOVED critical monitoring"""
        try:
            logger.info("Force running scheduled tasks...")
            
            self.schedule_anomaly_detection()
            time.sleep(1)
            
            self.schedule_predictions()
            time.sleep(1)
            
            self.schedule_model_training()
            
            logger.info("All tasks force-scheduled successfully")
            
        except Exception as e:
            logger.error(f"Error force running tasks: {e}")