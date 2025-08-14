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

        date_time_start = datetime.fromisoformat(task['date_time_start'])
        date_time_end = datetime.fromisoformat(task['date_time_end'])
        
        try:
            logger.info(f"Starting anomaly detection for aquarium {aquarium_id} from {date_time_start.isoformat()} to {date_time_end.isoformat()}")

            # Fetch anomaly settings
            anomaly_settings = self.supabase.get_aquarium_model_settings(aquarium_id, columns=['anomaly_parameters', 'contamination_rate'])

            if not anomaly_settings:
                logger.warning(f"No anomaly settings found for aquarium {aquarium_id}. Skipping anomaly detection.")
                return
            
            parameters = anomaly_settings.get('anomaly_parameters', ['ph', 'do', 'water_temperature'])
            contamination_rate = anomaly_settings.get('contamination_rate', 0.01)

            # Fetch historical data
            historical_data = self.supabase.get_historical_data(
                aquarium_id=aquarium_id,
                start_date= date_time_start,
                end_date=date_time_end
            )

            if historical_data.empty:
                logger.warning(f"No historical data found for aquarium {aquarium_id} in the specified range. Skipping anomaly detection.")
                return
            
            if len(historical_data) < 280:
                logger.warning(f"Not enough historical data for aquarium {aquarium_id} to perform anomaly detection. Minimum required is 280 records. {len(historical_data)} records found.")
                return
            for param in parameters:
                historical_data[f"{param}_change"] = historical_data[param].diff().abs()
                historical_data[f"{param}_rolling_mean"] = historical_data[param].rolling(window=5, center=True).mean()
                historical_data[f"{param}_rolling_std"] = historical_data[param].rolling(window=5, center=True).std()
                historical_data[f"{param}_deviation"] = abs(historical_data[param] - historical_data[f"{param}_rolling_mean"])

                # Fill NaN values
                historical_data[f"{param}_change"].dropna(inplace=True)
                historical_data.fillna({
                    f"{param}_rolling_mean": historical_data[param].mean(),
                    f"{param}_rolling_std": historical_data[param].std(),
                    f"{param}_deviation": 0
                }, inplace=True)

            anomalies = pd.DataFrame(columns=['datetime', 'parameter', 'anomaly_score', 'value', 'reason', 'severity'])

            for param in parameters:
                if param not in historical_data.columns:
                    logger.warning(f"Parameter {param} not found in historical data for aquarium {aquarium_id}. Skipping anomaly detection for this parameter.")
                    continue

                new_anomalies = self.ml_pipeline.detect_anomalies(
                    raw_data=historical_data,
                    parameter=param,
                    contamination=contamination_rate
                )

                if not new_anomalies.empty:
                    anomalies = pd.concat([anomalies, new_anomalies], ignore_index=True)

            # Insert anomalies into the database
            if not anomalies.empty:
                self.supabase.insert_anomalies(aquarium_id, anomalies)

            self.supabase.log_ml_activity(
                aquarium_id=aquarium_id,
                activity_type='anomaly_detection',
                status='completed',
                metadata={
                    'parameters': parameters,
                    'contamination_rate': contamination_rate,
                    'anomalies_detected': len(anomalies),
                    'date_time_start': date_time_start.isoformat(),
                    'date_time_end': date_time_end.isoformat()
                }
            )

            logger.info(f"Anomaly detection completed for aquarium {aquarium_id}. Detected {len(anomalies)} anomalies.")

            # Send alerts for detected anomalies

            # Fetch aquarium data
            aquarium_data = self.supabase.get_aquarium_data(aquarium_id=aquarium_id, columns=['user_id', 'name', 'timezone'])

            if aquarium_data is not None:
                user_id = aquarium_data['user_id']
                aquarium_name = aquarium_data['name']

                # convert date_time_start and date_time_end to the aquarium's timezone
                aquarium_timezone = aquarium_data['timezone']
                date_time_start = date_time_start.astimezone(aquarium_timezone)
                date_time_end = date_time_end.astimezone(aquarium_timezone)

                # Convert date_time to human-readable format
                date_time_start = date_time_start.strftime('%Y-%m-%d %H:%M:%S')
                date_time_end = date_time_end.strftime('%Y-%m-%d %H:%M:%S')

                self.supabase.send_alert(
                    user_id=user_id,
                    title=f"Anomaly Detection Alert for {aquarium_name}",
                    message=f"Anomalies detected in aquarium {aquarium_name} from {date_time_start} to {date_time_end}.",
                    severity='high'
                )

        except Exception as e:
            logger.error(f"Error in anomaly detection for aquarium {aquarium_id}: {e}")
            raise
        
    def handle_predictions(self, task: dict):
        """Handle prediction tasks"""
        aquarium_id = task['aquarium_id']
        date_time_now = datetime.fromisoformat(task['date_time_now']).replace(minute=0, second=0, microsecond=0)
        
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
        date_time_start = datetime.fromisoformat(task['date_time_start'])
        date_time_end = datetime.fromisoformat(task['date_time_end'])

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