"""
Load testing script for API performance validation
Measures throughput and latency under concurrent load
"""
import requests
import time
import concurrent.futures
import statistics
from datetime import datetime

API_URL = "http://localhost:8000"

# Test data
TEST_PRODUCTS = [
    {
        "product_name": "Wireless Headphones",
        "category": "Electronics",
        "description": "Noise-cancelling Bluetooth headphones with 30-hour battery"
    },
    {
        "product_name": "Yoga Mat",
        "category": "Sports & Fitness",
        "description": "Non-slip eco-friendly mat with extra cushioning"
    },
    {
        "product_name": "Coffee Maker",
        "category": "Home & Kitchen",
        "description": "Programmable coffee maker with auto shut-off"
    },
    {
        "product_name": "Running Shoes",
        "category": "Footwear",
        "description": "Lightweight breathable shoes with cushioned sole"
    },
    {
        "product_name": "Smart Watch",
        "category": "Wearables",
        "description": "Fitness tracker with GPS and heart rate monitor"
    }
]


def make_request(product_index):
    """Make a single request to the API"""
    product = TEST_PRODUCTS[product_index % len(TEST_PRODUCTS)]
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json=product,
            timeout=30
        )
        latency = time.time() - start_time
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency": latency,
            "error": None
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "status_code": 0,
            "latency": latency,
            "error": str(e)
        }


def run_load_test(num_requests=100, max_workers=10):
    """Run load test with concurrent requests"""
    print("="*70)
    print("🔥 AD GENERATOR API - LOAD TEST")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   • Total requests: {num_requests}")
    print(f"   • Concurrent workers: {max_workers}")
    print(f"   • Target API: {API_URL}")
    print(f"\n🚀 Starting load test...")
    print("-"*70)
    
    results = []
    start_time = time.time()
    
    # Run concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        
        # Collect results with progress
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Progress: {i}/{num_requests} requests completed")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successful]
    
    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    
    # Success rate
    success_rate = len(successful) / len(results) * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}% ({len(successful)}/{len(results)})")
    
    if failed:
        print(f"❌ Failed Requests: {len(failed)}")
        for f in failed[:5]:  # Show first 5 failures
            print(f"   - Status {f['status_code']}: {f['error']}")
    
    # Throughput
    throughput = len(successful) / total_time
    print(f"\n🔥 Throughput: {throughput:.2f} requests/second")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    
    # Latency stats
    if latencies:
        print(f"\n📊 Latency Statistics:")
        print(f"   • Min: {min(latencies)*1000:.0f}ms")
        print(f"   • Max: {max(latencies)*1000:.0f}ms")
        print(f"   • Mean: {statistics.mean(latencies)*1000:.0f}ms")
        print(f"   • Median (p50): {statistics.median(latencies)*1000:.0f}ms")
        
        # Percentiles
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        print(f"   • p95: {p95*1000:.0f}ms")
        print(f"   • p99: {p99*1000:.0f}ms")
    
    print("\n" + "="*70)
    
    # Recommendations
    print("\n💡 Analysis:")
    if success_rate >= 95:
        print("   ✅ Excellent success rate!")
    elif success_rate >= 80:
        print("   ⚠️  Good, but some failures detected")
    else:
        print("   ❌ High failure rate - investigate errors")
    
    if latencies and statistics.mean(latencies) < 2.0:
        print("   ✅ Good average latency")
    elif latencies:
        print("   ⚠️  High latency - consider optimization")
    
    if throughput >= 1.0:
        print(f"   ✅ Decent throughput ({throughput:.1f} req/s)")
    else:
        print(f"   ⚠️  Low throughput ({throughput:.1f} req/s)")
    
    print("="*70)
    
    return {
        "success_rate": success_rate,
        "throughput": throughput,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "p99_latency": p99 if latencies else 0
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test the Ad Generator API")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    
    args = parser.parse_args()
    
    API_URL = args.url
    
    # Run test
    results = run_load_test(num_requests=args.requests, max_workers=args.workers)
