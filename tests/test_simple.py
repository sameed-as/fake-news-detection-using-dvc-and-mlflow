"""
Simple API tests that work without complex imports
"""
import requests
import time

API_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health endpoint"""
    print("\n🧪 Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] == True
    print("   ✅ Health check passed")

def test_generate_endpoint():
    """Test generate endpoint"""
    print("\n🧪 Testing /generate endpoint...")
    response = requests.post(f"{API_URL}/generate", json={
        "product_name": "Test Product",
        "category": "Electronics",
        "description": "A test product"
    })
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "ad_creative" in data
    assert "quality_score" in data
    assert len(data["ad_creative"]) > 0
    print(f"   ✅ Generated ad: {data['ad_creative'][:50]}...")

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print("\n🧪 Testing /metrics endpoint...")
    response = requests.get(f"{API_URL}/metrics")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "text/plain" in response.headers["content-type"]
    assert len(response.text) > 100
    print("   ✅ Metrics endpoint working")

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("🧪 RUNNING API TESTS")
    print("="*70)
    
    try:
        test_health_endpoint()
        test_generate_endpoint()
        test_metrics_endpoint()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()
