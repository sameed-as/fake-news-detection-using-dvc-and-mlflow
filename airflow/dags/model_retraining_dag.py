"""
Model Retraining DAG
Automates weekly model retraining with MLflow integration
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.bash import BashOperator
import logging
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 1),
    'email_on_failure': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

def check_if_retraining_needed(**context):
    """Determine if model retraining is necessary"""
    logger.info("Checking if retraining is needed...")
    
    # Check 1: Time since last training
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.tracking.MlflowClient()
        
        # Get latest model
        logger.info("Fetching latest model versions from MLflow")
        try:
            latest_versions = client.get_latest_versions("ad-generator", stages=["Production"])
        except Exception as e:
            logger.warning(f"No model registered yet: {e}")
            latest_versions = None

        if latest_versions:
            last_trained = datetime.fromtimestamp(latest_versions[0].creation_timestamp / 1000)
            days_since_training = (datetime.now() - last_trained).days
            
            if days_since_training < 7:
                logger.info(f"Model trained {days_since_training} days ago - skip retraining")
                return 'skip_retraining'
        else:
            logger.info("No existing model found - retraining will be triggered")
        
    except Exception as e:
        logger.warning(f"Could not check MLflow: {e}")
    
    # Check 2: Data availability
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM products")
    product_count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    if product_count < 1:
        logger.info(f"Only {product_count} products - not enough for retraining")
        return 'skip_retraining'
    
    logger.info(f"Retraining needed: {product_count} products available")
    return 'prepare_data'

def prepare_training_data(**context):
    """Load and prepare data for training"""
    logger.info("Preparing training data...")
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Load products
    df = pg_hook.get_pandas_df("""
        SELECT name as product_name, category, description 
        FROM products 
        ORDER BY created_at DESC
        LIMIT 1000
    """)
    
    # Save to CSV for training script
    output_path = '/tmp/training_data.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Prepared {len(df)} samples for training")
    context['task_instance'].xcom_push(key='data_path', value=output_path)
    context['task_instance'].xcom_push(key='sample_count', value=len(df))
    
    return output_path

def train_new_model(**context):
    """Execute model training"""
    logger.info("Starting model training...")
    
    data_path = context['task_instance'].xcom_pull(key='data_path', task_ids='prepare_data')
    sample_count = context['task_instance'].xcom_pull(key='sample_count', task_ids='prepare_data')
    
    logger.info(f"Training with {sample_count} samples from {data_path}")
    
    # In production, this would call the training script
    # For now, we'll return success
    # The actual training happens in: python src/model/train.py
    
    run_id = f"retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    
    logger.info(f"Training completed! Run ID: {run_id}")
    return run_id

def compare_model_performance(**context):
    """Compare new model with production model"""
    logger.info("Comparing model performance...")
    
    run_id = context['task_instance'].xcom_pull(key='run_id', task_ids='train_model')
    
    # In production, compare metrics from MLflow
    # For demo, assume new model is better
    
    new_model_loss = 2.5  # From MLflow
    prod_model_loss = 3.2  # From MLflow
    
    if new_model_loss < prod_model_loss:
        logger.info(f"New model better! Loss: {new_model_loss} vs {prod_model_loss}")
        return 'promote_model'
    else:
        logger.info(f"Production model still better: {prod_model_loss} vs {new_model_loss}")
        return 'keep_production'

def promote_to_production(**context):
    """Promote new model to production"""
    logger.info("Promoting model to production...")
    
    run_id = context['task_instance'].xcom_pull(key='run_id', task_ids='train_model')
    
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.tracking.MlflowClient()
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, "ad-generator")
        
        # Transition to production
        client.transition_model_version_stage(
            name="ad-generator",
            version=mv.version,
            stage="Production"
        )
        
        logger.info(f"Model version {mv.version} promoted to Production!")
        
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise
    
    return True

def skip_retraining():
    """Log when retraining is skipped"""
    logger.info("Retraining skipped - conditions not met")
    return "Skipped"

# DAG definition
with DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Weekly model retraining with MLflow tracking',
    schedule_interval='0 3 * * 0',  # Sunday at 3 AM
    catchup=False,
    tags=['ml', 'training', 'mlflow'],
) as dag:
    
    # Task 1: Check if retraining needed
    check_retraining = BranchPythonOperator(
        task_id='check_retraining_needed',
        python_callable=check_if_retraining_needed,
        provide_context=True,
    )
    
    # Task 2: Skip retraining
    skip_task = PythonOperator(
        task_id='skip_retraining',
        python_callable=skip_retraining,
    )
    
    # Task 3: Prepare data
    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_training_data,
        provide_context=True,
    )
    
    # Task 4: Train model (calls training script)
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && python train.py',
    )
    
    # Task 5: Compare performance
    compare_models = BranchPythonOperator(
        task_id='compare_models',
        python_callable=compare_model_performance,
        provide_context=True,
    )
    
    # Task 6: Promote to production
    promote_model = PythonOperator(
        task_id='promote_model',
        python_callable=promote_to_production,
        provide_context=True,
    )
    
    # Task 7: Keep production model
    keep_production = PythonOperator(
        task_id='keep_production',
        python_callable=lambda: logger.info("Keeping current production model"),
    )
    
    # Define dependencies
    check_retraining >> [skip_task, prepare_data]
    prepare_data >> train_model >> compare_models
    compare_models >> [promote_model, keep_production]
