import pika
import json
import logging
import uuid
from typing import Dict, Any, Optional, Callable
from src.config.settings import settings
from datetime import datetime


logger = logging.getLogger(__name__)

class RabbitMQManager:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchange_name = 'aquawatch_ml'

    def connect(self) -> bool:
        try: 
            parameters = pika.URLParameters(settings.RABBITMQ_URL)
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare the exchange
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type='topic',
                durable=True
            )

            # Setup queues
            self.setup_queues()

            logger.info("Connected to RabbitMQ successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False

    def setup_queues(self):
        """Setup task queues"""

        queues = {
            'anomaly_detection': {'priority': 200, 'ttl': 1800000},     # 30 minutes TTL
            'predictions': {'priority': 150, 'ttl': 1800000},          # 30 minutes TTL
            'model_training': {'priority': 100, 'ttl': 3600000},        # 1 hour TTL
            'validate_predictions': {'priority': 50, 'ttl': 3600000}  # 1 hour TTL
        }

        for queue_name, config in queues.items():
            self.channel.queue_declare( # type: ignore
                queue=queue_name,
                durable=True,
                arguments={
                    'x-max-priority': 255,
                    'x-message-ttl': config['ttl']
                }
            )
            
            self.channel.queue_bind( # type: ignore
                exchange=self.exchange_name,
                queue=queue_name,
                routing_key=f"task.{queue_name}"
            )

        logger.info("Queues setup completed.")

    def publish_task(self, queue_name: str, payload: Dict[str, Any]):
        """
        Publish a task to the specified queue.

        Args:
            queue_name (str): Name of the queue to publish to.
            payload (Dict[str, Any]): The task payload.
        """

        try:
            attempted = 0
            max_attempts = 3

            if not self.channel:
                raise Exception("Channel is not initialized. Call connect() first.")

            # Check channel health
            if self.channel.is_closed:
                logger.info("Channel is closed, creating a new connection.")
                while attempted < max_attempts:
                    try:
                        status = self.connect()
                        if status:
                            logger.info("Reconnected to RabbitMQ successfully.")
                            break
                        attempted += 1
                    except Exception as e:
                        logger.error(f"Attempt {attempted + 1} failed: {e}")
                        attempted += 1

            if attempted == max_attempts:
                raise Exception("Failed to reconnect to RabbitMQ after multiple attempts.")

            task = {
                'task_id': str(uuid.uuid4()),
                'created_at': datetime.now().isoformat(),
                **payload
            }

            self.channel.basic_publish(
                exchange=self.exchange_name,
                routing_key=f"task.{queue_name}",
                body=json.dumps(task),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    message_id=task['task_id'],
                    priority=payload.get('priority', 128)
                )
            )

            logger.info(f"Published task to {queue_name}: {task['task_id']}")
            
        except Exception as e:
            logger.error(f"Failed to publish task: {e}")

    def close(self):
        """Close connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def create_direct_connection(self):
        """Create a new direct connection for workers"""
        try:
            parameters = pika.URLParameters(settings.RABBITMQ_URL)
            connection = pika.BlockingConnection(parameters)
            return connection
        except Exception as e:
            logger.error(f"Failed to create direct connection: {e}")
            raise

    def get_queue_info(self, queue_name: str) -> Dict:
        """Get information about a queue"""
        try:
            if not self.channel:
                raise Exception("Channel is not initialized. Call connect() first.")

            method = self.channel.queue_declare(queue=queue_name, passive=True)
            return {
                'queue': queue_name,
                'message_count': method.method.message_count,
                'consumer_count': method.method.consumer_count
            }
        except Exception as e:
            logger.error(f"Failed to get queue info for {queue_name}: {e}")
            return {}
    
    def purge_queue(self, queue_name: str) -> bool:
        """Purge all messages from a queue"""
        try:
            if not self.channel:
                raise Exception("Channel is not initialized. Call connect() first.")

            self.channel.queue_purge(queue=queue_name)
            logger.info(f"Purged queue: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return False
    
    def get_all_queue_stats(self) -> Dict:
        """Get stats for all queues - REMOVED critical_monitoring"""
        queues = ['anomaly_detection', 'predictions', 'model_training', 'validate_predictions']  # Removed critical_monitoring
        stats = {}
        
        for queue_name in queues:
            stats[queue_name] = self.get_queue_info(queue_name)
        
        return stats
