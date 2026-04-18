"""
MinIO bucket creation and verification script
"""
from minio import Minio
from minio.error import S3Error

# MinIO client configuration
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "mlflow"

print(f"Checking if bucket '{bucket_name}' exists...")

try:
    # Check if bucket exists
    if client.bucket_exists(bucket_name):
        print(f"✅ Bucket '{bucket_name}' already exists")
    else:
        # Create bucket
        client.make_bucket(bucket_name)
        print(f"✅ Bucket '{bucket_name}' created successfully")
    
    # List all buckets
    buckets = client.list_buckets()
    print(f"\n📦 Available buckets:")
    for bucket in buckets:
        print(f"  - {bucket.name} (created: {bucket.creation_date})")
        
except S3Error as e:
    print(f"❌ Error: {e}")
