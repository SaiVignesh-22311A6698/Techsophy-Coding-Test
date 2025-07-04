#!/usr/bin/env python3
"""
Test script for the EHR Data Quality Auditor Flask backend
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_sample_data():
    """Test the sample data endpoint"""
    print("\nTesting sample data endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/sample-data")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Record count: {data.get('record_count', 0)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analyze():
    """Test the analyze endpoint with sample data"""
    print("\nTesting analyze endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/analyze", json={})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Overall quality score: {data.get('overall_quality_score', 'N/A')}")
            print(f"Total records: {data.get('total_records', 'N/A')}")
            print(f"Processing time: {data.get('processing_time', 'N/A')} seconds")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_metrics():
    """Test the metrics endpoint"""
    print("\nTesting metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Completeness threshold: {data.get('completeness_metrics', {}).get('completeness_threshold', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== EHR Data Quality Auditor Backend Test ===\n")
    
    tests = [
        ("Health Check", test_health),
        ("Sample Data", test_sample_data),
        ("Analyze Data", test_analyze),
        ("Quality Metrics", test_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
        print("-" * 50)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the backend.")

if __name__ == "__main__":
    main() 