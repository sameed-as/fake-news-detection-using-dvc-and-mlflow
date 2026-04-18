"""
Batch Inference DAG
Generates ads in batch for large product catalogs
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import logging
import requests

logger = logging.getLogger(__name__)

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 1),
    'email_on_failure': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=1),
}

def load_products_needing_ads(**context):
    """Load products that need ad generation"""
    logger.info("Loading products needing ads...")
    
    pg_hook = PostgresHook(postgres_conn_id='airflow-postgres')
    
    # Get products without ads
    df = pg_hook.get_pandas_df("""
        SELECT id, name as product_name, category, description 
        FROM products 
        WHERE id NOT IN (SELECT product_id FROM generated_ads)
        LIMIT 100
    """)
    
    logger.info(f"Found {len(df)} products needing ads")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='products', value=df.to_dict('records'))
    context['task_instance'].xcom_push(key='count', value=len(df))
    
    return len(df)

def generate_ads_batch(**context):
    """Generate ads using the API"""
    logger.info("Generating ads in batch...")
    
    products = context['task_instance'].xcom_pull(key='products', task_ids='load_products')
    
    if not products:
        logger.info("No products to process")
        return 0
    
    API_URL = "http://api:8000/generate"
    
    generated_ads = []
    successful = 0
    
    for product in products:
        try:
            # Call API
            response = requests.post(API_URL, json={
                "product_name": product['product_name'],
                "category": product['category'],
                "description": product['description']
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_ads.append({
                    'product_id': product['id'],
                    'product_name': product['product_name'],
                    'ad_text': result['ad_text'],
                    'model_version': result.get('model_version', 'unknown'),
                    'confidence': result.get('quality_score', 0.0)
                })
                successful += 1
            else:
                logger.error(f"API error for {product['product_name']}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error generating ad for {product['product_name']}: {e}")
    
    logger.info(f"Successfully generated {successful}/{len(products)} ads")
    
    # Push results
    context['task_instance'].xcom_push(key='generated_ads', value=generated_ads)
    context['task_instance'].xcom_push(key='successful_count', value=successful)
    
    return successful

def store_generated_ads(**context):
    """Store generated ads in database"""
    logger.info("Storing generated ads...")
    
    ads = context['task_instance'].xcom_pull(key='generated_ads', task_ids='generate_ads')
    
    if not ads:
        logger.info("No ads to store")
        return 0
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_ads (
            id SERIAL PRIMARY KEY,
            product_id INTEGER REFERENCES products(id),
            ad_text TEXT NOT NULL,
            model_version VARCHAR(50),
            confidence FLOAT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert ads
    inserted = 0
    for ad in ads:
        try:
            cursor.execute("""
                INSERT INTO generated_ads (product_id, ad_text, model_version, confidence)
                VALUES (%s, %s, %s, %s)
            """, (ad['product_id'], ad['ad_text'], ad['model_version'], ad['confidence']))
            inserted += cursor.rowcount
        except Exception as e:
            logger.error(f"Error inserting ad: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"Stored {inserted} ads in database")
    return inserted

def export_to_csv(**context):
    """Export generated ads to CSV"""
    logger.info("Exporting ads to CSV...")
    
    ads = context['task_instance'].xcom_pull(key='generated_ads', task_ids='generate_ads')
    
    if not ads:
        logger.info("No ads to export")
        return None
    
    df = pd.DataFrame(ads)
    
    # Export
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/tmp/generated_ads_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(df)} ads to {output_path}")
    
    return output_path

def quality_check_ads(**context):
    """Perform quality checks on generated ads"""
    logger.info("Performing quality checks...")
    
    ads = context['task_instance'].xcom_pull(key='generated_ads', task_ids='generate_ads')
    
    if not ads:
        return True
    
    issues = []
    
    for ad in ads:
        # Check 1: Minimum length
        if len(ad['ad_text']) < 20:
            issues.append(f"{ad['product_name']}: Ad too short")
        
        # Check 2: Maximum length
        if len(ad['ad_text']) > 500:
            issues.append(f"{ad['product_name']}: Ad too long")
        
        # Check 3: Contains product name
        if ad['product_name'].lower() not in ad['ad_text'].lower():
            issues.append(f"{ad['product_name']}: Product name not in ad")
        
        # Check 4: Confidence threshold
        if ad['confidence'] < 0.3:
            issues.append(f"{ad['product_name']}: Low confidence ({ad['confidence']:.2f})")
    
    if issues:
        logger.warning(f"Found {len(issues)} quality issues:")
        for issue in issues[:10]:  # Log first 10
            logger.warning(f"  - {issue}")
    else:
        logger.info("All ads passed quality checks!")
    
    # Push metrics
    context['task_instance'].xcom_push(key='quality_issues', value=len(issues))
    context['task_instance'].xcom_push(key='pass_rate', value=(len(ads) - len(issues)) / len(ads))
    
    return len(issues) == 0

# DAG definition  
with DAG(
    'batch_inference_pipeline',
    default_args=default_args,
    description='Generate ads for product catalog',
    schedule_interval='0 4 * * 1',  #Every Monday at 4 AM
    catchup=False,
    tags=['inference', 'batch', 'ads'],
) as dag:
    
    # Task 1: Load products
    load_products = PythonOperator(
        task_id='load_products',
        python_callable=load_products_needing_ads,
        provide_context=True,
    )
    
    # Task 2: Generate ads
    generate_ads = PythonOperator(
        task_id='generate_ads',
        python_callable=generate_ads_batch,
        provide_context=True,
    )
    
    # Task 3: Quality check
    quality_check = PythonOperator(
        task_id='quality_check',
        python_callable=quality_check_ads,
        provide_context=True,
    )
    
    # Task 4: Store in database
    store_ads = PythonOperator(
        task_id='store_ads',
        python_callable=store_generated_ads,
        provide_context=True,
    )
    
    # Task 5: Export to CSV
    export_csv = PythonOperator(
        task_id='export_csv',
        python_callable=export_to_csv,
        provide_context=True,
    )
    
    # Define dependencies
    load_products >> generate_ads >> quality_check >> [store_ads, export_csv]
