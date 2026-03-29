"""Test script for CreditIQ API."""

import json

import httpx

# Test application data
TEST_APPLICATION = {
    "age_years": 35,
    "gender": "M",
    "family_members": 2,
    "children_count": 0,
    "income_total": 180000,
    "credit_amount": 450000,
    "annuity_amount": 22500,
    "goods_price": 450000,
    "employment_years": 5,
    "income_type": "Working",
    "organization_type": "Business Entity Type 3",
    "ext_source_1": 0.6,
    "ext_source_2": 0.7,
    "ext_source_3": 0.65,
    "own_car": True,
    "own_realty": True,
    "car_age": 5.0,
    "education_type": "Higher education",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "loan_purpose": "Home renovation for family property",
    "contract_type": "Cash loans",
}

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n✓ Testing health endpoint...")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200


def test_root():
    """Test root endpoint."""
    print("\n✓ Testing root endpoint...")
    response = httpx.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200


def test_single_assessment():
    """Test single loan assessment."""
    print("\n✓ Testing single assessment endpoint...")
    response = httpx.post(f"{BASE_URL}/assess/", json=TEST_APPLICATION, timeout=30.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert "credit_score" in result
    assert "default_probability" in result
    assert "recommendation" in result
    assert "top_risk_factors" in result
    assert "top_protective_factors" in result

    print(f"\n  Credit Score: {result['credit_score']}")
    print(f"  Risk Band: {result['risk_band']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Default Probability: {result['default_probability']:.4f}")
    print(f"  Inference Time: {result['inference_time_ms']}ms")


def test_batch_assessment():
    """Test batch assessment."""
    print("\n✓ Testing batch assessment endpoint...")

    # Create 3 test applications with slight variations
    batch_data = {
        "applications": [
            TEST_APPLICATION,
            {**TEST_APPLICATION, "age_years": 28, "income_total": 120000},
            {**TEST_APPLICATION, "age_years": 50, "income_total": 250000},
        ]
    }

    response = httpx.post(f"{BASE_URL}/assess/batch", json=batch_data, timeout=60.0)
    print(f"Status: {response.status_code}")
    result = response.json()

    print("\nBatch Summary:")
    print(f"  Total Applications: {result['total']}")
    print(f"  Processed: {result['processed']}")
    print(f"  Average Score: {result['summary']['average_score']:.2f}")
    print(f"  Min Score: {result['summary']['min_score']}")
    print(f"  Max Score: {result['summary']['max_score']}")
    print(f"  Approve Count: {result['summary']['approve_count']}")
    print(f"  Decline Count: {result['summary']['decline_count']}")
    print(f"  Review Count: {result['summary']['review_count']}")

    assert response.status_code == 200
    assert result["total"] == 3
    assert result["processed"] == 3


def test_edge_cases():
    """Test edge cases."""
    print("\n✓ Testing edge cases...")

    # Test with minimal ext_source values
    edge_case = {
        **TEST_APPLICATION,
        "ext_source_1": None,
        "ext_source_2": None,
        "ext_source_3": None,
        "employment_years": None,
        "car_age": None,
    }

    response = httpx.post(f"{BASE_URL}/assess/", json=edge_case, timeout=30.0)
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    result = response.json()
    print(f"  Credit Score: {result['credit_score']}")
    print(f"  Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    print("=" * 70)
    print("CreditIQ API Test Suite")
    print("=" * 70)

    try:
        test_health()
        test_root()
        test_single_assessment()
        test_batch_assessment()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    except httpx.ConnectError:
        print("\n✗ Could not connect to API at {BASE_URL}")
        print("  Make sure the API is running with: uvicorn api.main:app --reload")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
