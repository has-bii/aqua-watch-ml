import json
from datetime import datetime, timezone
from src.libs.supabase_manager import SupabaseManager
from src.libs.rabbitmq_manager import RabbitMQManager
from src.libs.ml_pipeline import MLPipeline
import pandas as pd
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
        date_time_now: datetime = task.get('date_time_now', datetime.now(timezone.utc))
        
        try:
            self.ml_pipeline.predict(
                aquarium_id=aquarium_id,
                date_time_now=date_time_now,
            )

            logger.info(f"Prediction task completed for aquarium {aquarium_id}")
            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type='predictions',
                status='completed',
            )
            
        except Exception as e:
            logger.error(f"Error in predictions for aquarium {aquarium_id}: {e}")
            raise
    
    def handle_model_training(self, task: dict):
        """Handle model training tasks"""
        aquarium_id = task['aquarium_id']
        parameters = task.get('parameters', ['water_temperature', 'ph', 'do'])
        tasked_at = task.get('tasked_at', datetime.now(timezone.utc))
        
        try:
            # Fetch aquarium model settings
            aquarium_settings = self.supabase.get_aquarium_model_settings(aquarium_id)

            if not aquarium_settings:
                raise ValueError("No model settings found for the aquarium.")

            # Get days back for training
            days_back = aquarium_settings['train_model_day_count']

            if not days_back or days_back <= 0:
                raise ValueError("Invalid train_model_day_count in aquarium settings.")

            # Calculate start and end dates
            start_date = (tasked_at - pd.Timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
            end_date = tasked_at

            # Fetch historical data
            historical_data = self.supabase.get_historical_data(aquarium_id, start_date=start_date, end_date=end_date)

            # Reindex to 15-minute intervals
            historical_data['created_at'] = pd.to_datetime(historical_data['created_at'], format='ISO8601')
            historical_data.set_index('created_at', inplace=True)
            historical_data.sort_index(inplace=True)

            full_index = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max(), freq='15min')
            historical_data = historical_data.reindex(full_index)
            for param in parameters:
                if param in historical_data.columns:
                    historical_data[param] = historical_data[param].interpolate(method='time').ffill().bfill()
            historical_data.index.name = 'created_at'
            historical_data.reset_index(inplace=True)

            # Fetch water change data
            water_change_data = self.supabase.get_water_changing_data(aquarium_id, start_date=start_date, end_date=end_date)

            # Train water temperature model
            result = self.ml_pipeline.train_water_temperature(
                aquarium_id=aquarium_id,
                historical_data=historical_data,
                water_change_data=water_change_data,
                days_back=int(days_back)
            )
            
            if not result['success']:
                raise RuntimeError(f"Model training failed: {result.get('error', 'Unknown error')}")
            
            # Train ph model
            # placeholder for ph model training

            
            
        except Exception as e:
            logger.error(f"Error in model training for aquarium {aquarium_id}: {e}")
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