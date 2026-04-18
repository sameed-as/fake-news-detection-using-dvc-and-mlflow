"""
Data Ingestion DAG
Automates daily product data collection and preparation
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def fetch_new_products(**context):
    """Simulate fetching new product data from external source"""
    logger.info("Fetching new product data...")
    
    # In production, this would call an API or read from S3/cloud storage
    # For demo, we'll create sample data
    now = datetime.now()
    logger.info(f"Current timestamp for new products: {now}")
    new_products = pd.DataFrame({
        'product_name': ['New Smart Watch', 'Wireless Charger Pro', 'Gaming Keyboard'],
        'category': ['Electronics', 'Electronics', 'Electronics'],
        'description': [
            'Advanced fitness tracking with heart rate and GPS',
            'Fast wireless charging for all devices',
            'Mechanical switches with RGB lighting'
        ],
        'date_added': now.isoformat()
    })
    
    logger.info(f"Fetched {len(new_products)} new products")
    
    # Push data to XCom for next task
    context['task_instance'].xcom_push(key='new_products', value=new_products.to_dict('records'))
    return len(new_products)

def validate_data_quality(**context):
    """Validate data quality before insertion"""
    logger.info("Validating data quality...")
    
    products = context['task_instance'].xcom_pull(key='new_products', task_ids='fetch_products')
    
    if not products:
        raise ValueError("No products to validate")
    
    # Quality checks
    for product in products:
        # Check required fields
        if not all(key in product for key in ['product_name', 'category', 'description']):
            raise ValueError(f"Missing required fields in product: {product}")
        
        # Check non-empty
        if not product['product_name'] or not product['description']:
            raise ValueError(f"Empty fields in product: {product}")
        
        # Check description length
        if len(product['description']) < 10:
            logger.warning(f"Short description for {product['product_name']}")
    
    logger.info(f"Validated {len(products)} products - all passed!")
    return True

def store_in_database(**context):
    """Store validated products in PostgreSQL"""
    logger.info("Storing products in database...")
    
    products = context['task_instance'].xcom_pull(key='new_products', task_ids='fetch_products')
    
    # Connect to PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Insert products
    inserted = 0
    for product in products:
        try:
            cursor.execute("""
                INSERT INTO products (name, category, description, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (
                product['product_name'],
                product['category'],
                product['description'],
                product.get('date_added', datetime.now())
            ))
            inserted += cursor.rowcount
        except Exception as e:
            logger.error(f"Error inserting product: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"Inserted {inserted} new products into database")
    return inserted

def check_retraining_trigger(**context):
    """Check if we should trigger model retraining"""
    logger.info("Checking if retraining needed...")
    
    inserted_count = context['task_instance'].xcom_pull(task_ids='store_products')
    
    # Trigger retraining if we have significant new data
    RETRAINING_THRESHOLD = 100
    
    if inserted_count >= RETRAINING_THRESHOLD:
        logger.info(f"Triggering retraining: {inserted_count} new products added!")
        # In production, trigger the retraining DAG
        return "trigger_retraining"
    else:
        logger.info(f"No retraining needed: only {inserted_count} products added")
        return "skip_retraining"

# DAG definition
with DAG(
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Daily data ingestion and validation',
    schedule_interval='0 0 * * *',  # Daily at 12 AM
    catchup=False,
    tags=['data', 'ingestion', 'etl'],
) as dag:
    
    # Task 1: Create table if not exists
    create_table = PostgresOperator(
        task_id='create_products_table',
        postgres_conn_id='postgres_default',
        sql="""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            category VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    
    # Task 2: Fetch new products
    fetch_products = PythonOperator(
        task_id='fetch_products',
        python_callable=fetch_new_products,
        provide_context=True,
    )
    
    # Task 3: Validate data quality
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    # Task 4: Store in database
    store_products = PythonOperator(
        task_id='store_products',
        python_callable=store_in_database,
        provide_context=True,
    )
    
    # Task 5: Check if retraining needed
    check_retraining = PythonOperator(
        task_id='check_retraining',
        python_callable=check_retraining_trigger,
        provide_context=True,
    )
    
    # Define task dependencies
    create_table >> fetch_products >> validate_data >> store_products >> check_retraining
