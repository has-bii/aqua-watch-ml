from src.config.settings import settings
import pika

def connect_rabbitmq():
    try:
        # Test basic connection
        connection = pika.BlockingConnection(
            pika.URLParameters(settings.RABBITMQ_URL)
        )
        print("Connection successful!")
        connection.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    connect_rabbitmq()
    print("RabbitMQ connection test completed.")