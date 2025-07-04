#!/usr/bin/env python3
"""
Test script to verify that infinite values are handled correctly in JSON serialization.
"""

import json
import numpy as np
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api import clean_data_for_json, NumpyJSONResponse

def test_infinite_values():
    """Test that infinite values are properly handled"""
    
    # Test data with various problematic values
    test_data = {
        "normal_float": 3.14,
        "infinite_pos": np.inf,
        "infinite_neg": -np.inf,
        "nan_value": np.nan,
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(2.71),
        "array_with_inf": np.array([1.0, np.inf, 3.0, -np.inf, np.nan]),
        "nested_dict": {
            "inner_inf": np.inf,
            "inner_normal": 1.23
        },
        "list_with_inf": [1.0, np.inf, 3.0]
    }
    
    print("Original data:")
    print(test_data)
    print("\n" + "="*50 + "\n")
    
    # Test our cleaning function
    cleaned_data = clean_data_for_json(test_data)
    print("Cleaned data:")
    print(cleaned_data)
    print("\n" + "="*50 + "\n")
    
    # Test JSON serialization
    try:
        json_str = json.dumps(cleaned_data)
        print("JSON serialization successful!")
        print("JSON string:", json_str[:100] + "..." if len(json_str) > 100 else json_str)
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        print("JSON deserialization successful!")
        
        return True
        
    except Exception as e:
        print(f"JSON serialization failed: {e}")
        return False

def test_stats_like_data():
    """Test with data similar to what causes the inference stats error"""
    
    # Simulate problematic inference stats data
    stats_data = {
        "model_1": [
            {
                "timestamp": "2025-01-01T12:00:00",
                "prediction_time": 0.5,
                "n_samples": 0,  # This causes division by zero -> inf
                "avg_time_per_sample": np.inf,  # This is the problematic value
                "accuracy": 0.95
            }
        ]
    }
    
    print("Stats-like data with inf:")
    print(stats_data)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_stats = clean_data_for_json(stats_data)
    print("Cleaned stats data:")
    print(cleaned_stats)
    print("\n" + "="*50 + "\n")
    
    # Test JSON serialization
    try:
        json_str = json.dumps(cleaned_stats)
        print("Stats JSON serialization successful!")
        return True
        
    except Exception as e:
        print(f"Stats JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing infinite value handling...\n")
    
    test1_result = test_infinite_values()
    print("\n" + "="*70 + "\n")
    test2_result = test_stats_like_data()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"Test 1 (General infinite values): {'PASSED' if test1_result else 'FAILED'}")
    print(f"Test 2 (Stats-like data): {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("\n✅ All tests passed! The infinite value handling should work correctly.")
    else:
        print("\n❌ Some tests failed. The fix may need adjustment.")
