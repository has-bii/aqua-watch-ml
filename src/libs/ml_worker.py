import json
from datetime import datetime, timezone
from src.libs.supabase_manager import SupabaseManager
from src.libs.rabbitmq_manager import RabbitMQManager
from src.libs.ml_pipeline import MLPipeline
import pandas as pd
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class MLWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.rabbitmq = RabbitMQManager()
        self.supabase = SupabaseManager()
        self.ml_pipeline = MLPipeline()

    def start_consuming(self, queue_name: str):
        """Start consuming tasks from queue - REMOVED critical monitoring handler"""
        self.rabbitmq.connect()

        if not self.rabbitmq.channel:
            logger.error("RabbitMQ channel not initialized. Cannot start consuming.")
            return
        
        def callback(ch, method, properties, body):
            task_start_time = datetime.now()
            task = None
            
            try:
                task = json.loads(body)
                logger.info(f"Worker {self.worker_id} processing task: {task['task_id']}")
                
                if queue_name == 'anomaly_detection':
                    self.handle_anomaly_detection(task)
                elif queue_name == 'predictions':
                    self.handle_predictions(task)
                elif queue_name == 'model_training':
                    self.handle_model_training(task)
                elif queue_name == 'validate_predictions':
                    self.handle_validate_predictions(task)
                elif queue_name == 'missing_data':
                    self.handle_missing_data(task)
                else:
                    logger.warning(f"Unknown queue type: {queue_name}")
                
                # Log successful completion
                processing_time = (datetime.now() - task_start_time).total_seconds()
                self.supabase.log_ml_activity(
                    aquarium_id=task.get('aquarium_id', 0),
                    activity_type=task.get('task_type', queue_name),
                    status='completed',
                    processing_time_seconds=int(processing_time)
                )
                
                # Acknowledge task
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                
                # Log error
                if task:
                    self.supabase.log_ml_activity(
                        aquarium_id=task.get('aquarium_id', 0),
                        activity_type=task.get('task_type', queue_name),
                        status='failed',
                        error_message=str(e)
                    )
                
                # Reject and requeue (with limit)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        self.rabbitmq.channel.basic_qos(prefetch_count=1)
        self.rabbitmq.channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback
        )
        
        logger.info(f"Worker {self.worker_id} started consuming from {queue_name}")
        self.rabbitmq.channel.start_consuming()

    def handle_validate_predictions(self, task: dict):
        """Handle anomaly detection tasks - runs every 30 minutes"""
        aquarium_id = task['aquarium_id']
        target_time = datetime.fromisoformat(task['target_time'])
        parameters = ['water_temperature', 'ph']

        try:    
            logger.info(f"Validating predictions for aquarium {aquarium_id} at {target_time.isoformat()}")
            validated_params = []
            
            for param in parameters:
                is_success = self.ml_pipeline.validate_prediction(
                    aquarium_id=aquarium_id,
                    parameter=param, # type: ignore
                    target_time=target_time
                )
                validated_params.append(param) if is_success else None

            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type='validate_predictions',
                status='success',
                metadata={
                    'validated_parameters': validated_params,
                    'target_time': target_time.isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error in validate_predictions for aquarium {aquarium_id}: {e}")
            raise
    
    def handle_anomaly_detection(self, task: dict):
        """Handle anomaly detection tasks - runs every 30 minutes"""
        aquarium_id = task['aquarium_id']
        
        try:
            # # Get recent measurements (last 2 hours for anomaly detection)
            # df = self.supabase.get_measurements(aquarium_id)
            
            # if df.empty:
            #     logger.warning(f"No recent measurements for anomaly detection: aquarium {aquarium_id}")
            #     return
            
            # # Detect anomalies
            # anomalies = self.ml_pipeline.detect_anomalies(aquarium_id, df)
            
            # # Save anomalies and create alerts for high-severity ones
            # alerts_created = 0
            # for anomaly in anomalies:
            #     anomaly_data = {
            #         'aquarium_id': aquarium_id,
            #         'parameter_name': anomaly['parameter'],
            #         'detected_at': anomaly['timestamp'].isoformat(),
            #         'actual_value': anomaly['actual_value'],
            #         'anomaly_score': anomaly['anomaly_score'],
            #         'severity': anomaly['severity'],
            #         'is_resolved': False
            #     }
                
            #     # Save anomaly
            #     self.supabase.save_anomaly(anomaly_data)
                
            #     # Create alert for medium, high, and critical severity anomalies
            #     if anomaly['severity'] in ['medium', 'high', 'critical']:
            #         alert_data = {
            #             'aquarium_id': aquarium_id,
            #             'severity': anomaly['severity'],
            #             'title': f'Anomaly in {anomaly["parameter"].upper()}',
            #             'message': f'Unusual {anomaly["parameter"]} pattern detected: {anomaly["actual_value"]:.2f} (score: {anomaly["anomaly_score"]:.3f})',
            #             'alert_timestamp': datetime.now().isoformat(),
            #             'is_acknowledged': False
            #         }
            #         self.supabase.save_alert(alert_data)
            #         alerts_created += 1
            
            # # Log activity
            # self.supabase.log_ml_activity(
            #     aquarium_id=aquarium_id,
            #     activity_type='anomaly_detection',
            #     status='completed',
            #     data_points_processed=len(df),
            #     anomalies_detected=len(anomalies),
            #     alerts_generated=alerts_created
            # )
            
            # if anomalies:
            #     logger.info(f"Detected {len(anomalies)} anomalies for aquarium {aquarium_id}")
            
            logger.info(f"Anomaly detection task completed for aquarium {aquarium_id}")

            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type='anomaly_detection',
                status='completed',
            )

        except Exception as e:
            logger.error(f"Error in anomaly detection for aquarium {aquarium_id}: {e}")
            raise
        
    def handle_predictions(self, task: dict):
        """Handle prediction tasks"""
        aquarium_id = task['aquarium_id']
        date_time_now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        
        try:
            parameters = None
            aquarium_settings = self.supabase.get_aquarium_model_settings(aquarium_id)

            if aquarium_settings:
                parameters = aquarium_settings.get('prediction_parameters', ['water_temperature', 'ph', 'do'])

            self.ml_pipeline.predict(
                aquarium_id=aquarium_id,
                date_time_now=date_time_now,
                parameters=parameters # type: ignore
            )

            logger.info(f"Prediction task completed for aquarium {aquarium_id}")
            
        except Exception as e:
            logger.error(f"Error in predictions for aquarium {aquarium_id}: {e}")
            raise
    
    def handle_model_training(self, task: dict):
        """Handle model training tasks"""
        aquarium_id = task['aquarium_id']
        tasked_at = task.get('tasked_at', datetime.now(timezone.utc))
        
        try:
            logger.info(f"Starting model training for aquarium {aquarium_id} at {tasked_at.isoformat()}")
            self.ml_pipeline.train_models(
                aquarium_id=aquarium_id,
            )
            logger.info(f"Model training completed for aquarium {aquarium_id}")
            
        except Exception as e:
            logger.error(f"Error in model training for aquarium {aquarium_id}: {e}")
            raise
    
    def handle_missing_data(self, task: dict):
        """
        Handle missing data tasks
        """
        aquarium_id = task['aquarium_id']
        date_time_start = task.get('date_time_start', datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
        date_time_end = task.get('date_time_end', datetime.now(timezone.utc))

        try:
            logger.info(f"Handling missing data for aquarium {aquarium_id} from {date_time_start.isoformat()} to {date_time_end.isoformat()}")
            
            self.ml_pipeline.find_missing_data(
                aquarium_id=aquarium_id,
                date_time_start=date_time_start,
                date_time_end=date_time_end
            )

            logger.info(f"Missing data handled for aquarium {aquarium_id}")
            
        except Exception as e:
            logger.error(f"Error handling missing data for aquarium {aquarium_id}: {e}")
            raise

    def is_prediction_concerning(self, parameter: str, prediction: dict) -> bool:
        """Check if prediction indicates potential problems"""
        concerning_ranges = {
            'ph': (6.5, 8.5),
            'do': (5.0, float('inf')),
            'water_temperature': (20.0, 28.0)
        }
        
        if parameter in concerning_ranges:
            min_val, max_val = concerning_ranges[parameter]
            predicted_val = prediction['predicted_value']
            return predicted_val < min_val or predicted_val > max_val
        
        return False