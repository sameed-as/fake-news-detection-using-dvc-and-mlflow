"""
Unit tests for Ad Generator API
Tests endpoints, model loading, and metrics
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'api')))

from main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_returns_200(self):
        """Health endpoint should return 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health endpoint should return proper JSON structure"""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "model_version" in data
    
    def test_health_model_loaded(self):
        """Model should be loaded at startup"""
        response = client.get("/health")
        data = response.json()
        
        assert data["model_loaded"] == True
        assert data["status"] == "healthy"


class TestGenerateEndpoint:
    """Test ad generation endpoint"""
    
    def test_generate_requires_product_data(self):
        """Generate endpoint should require product data"""
        response = client.post("/generate", json={})
        # Should fail validation
        assert response.status_code == 422
    
    def test_generate_with_valid_input(self):
        """Generate endpoint should work with valid input"""
        response = client.post("/generate", json={
            "product_name": "Test Product",
            "category": "Electronics",
            "description": "A test product for testing"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ad_creative" in data
        assert "quality_score" in data
        assert "generation_time" in data
        assert "model_version" in data
    
    def test_generate_returns_text(self):
        """Generated ad should be non-empty text"""
        response = client.post("/generate", json={
            "product_name": "Smart Watch",
            "category": "Wearables",
            "description": "Fitness tracker with GPS"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["ad_creative"]) > 0
        assert isinstance(data["ad_creative"], str)
    
    def test_generate_quality_score_range(self):
        """Quality score should be between 0 and 1"""
        response = client.post("/generate", json={
            "product_name": "Wireless Headphones",
            "category": "Electronics",
            "description": "Noise-cancelling headphones"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert 0.0 <= data["quality_score"] <= 1.0
    
    def test_generate_with_optional_price(self):
        """Generate should work with optional price field"""
        response = client.post("/generate", json={
            "product_name": "Coffee Maker",
            "category": "Home & Kitchen",
            "description": "Programmable coffee maker",
            "price": 99.99
        })
        
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""
    
    def test_metrics_returns_200(self):
        """Metrics endpoint should return 200 OK"""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_content_type(self):
        """Metrics should return Prometheus format"""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_contains_custom_metrics(self):
        """Metrics should include our custom metrics"""
        response = client.get("/metrics")
        content = response.text
        
        # Check for our custom metrics
        assert "requests_total" in content or "ad_generation" in content
        assert "HELP" in content  # Prometheus format
        assert "TYPE" in content  # Prometheus format


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_returns_200(self):
        """Root endpoint should return 200 OK"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_shows_endpoints(self):
        """Root should list available endpoints"""
        response = client.get("/")
        data = response.json()
        
        assert "endpoints" in data
        assert "generate" in data["endpoints"]
        assert "health" in data["endpoints"]
        assert "metrics" in data["endpoints"]


class TestAPIIntegration:
    """Integration tests"""
    
    def test_full_generation_workflow(self):
        """Test complete workflow: health check -> generate -> metrics"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] == True
        
        # 2. Generate ad
        gen_response = client.post("/generate", json={
            "product_name": "Test Product",
            "category": "Test",
            "description": "Test description"
        })
        assert gen_response.status_code == 200
        assert len(gen_response.json()["ad_creative"]) > 0
        
        # 3. Check metrics updated
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert len(metrics_response.text) > 100  # Should have metrics data
    
    def test_multiple_generations(self):
        """Test generating multiple ads in sequence"""
        products = [
            {"product_name": "Product A", "category": "Cat A", "description": "Desc A"},
            {"product_name": "Product B", "category": "Cat B", "description": "Desc B"},
            {"product_name": "Product C", "category": "Cat C", "description": "Desc C"},
        ]
        
        for product in products:
            response = client.post("/generate", json=product)
            assert response.status_code == 200
            assert "ad_creative" in response.json()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
