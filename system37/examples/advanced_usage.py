#!/usr/bin/env python3

from system37 import PatternEngine, Base37SymbolSystem, ValidationFramework
import numpy as np
from datetime import datetime

def matrix_transformation_example():
    """Advanced matrix transformation example"""
    print("\n=== Advanced Matrix Transformation ===")
    
    engine = PatternEngine()
    
    # Process pattern with matrix transformation
    pattern = "COMPLEX123"
    result = engine.process_pattern(pattern, secure=True)
    
    # Access matrix data
    matrix = result['transformation']['matrix']
    
    print(f"Input Pattern: {pattern}")
    print("\nSpiral Matrix:")
    for row in matrix:
        print(' '.join(bit if bit else '0' for bit in row))
        
    print(f"\nXOR Result: {result['transformation']['xored'][:32]}...")
    print(f"Final Output: {result['transformation']['final'][:32]}...")

def custom_phase_example():
    """Custom binary phase processing example"""
    print("\n=== Custom Phase Processing ===")
    
    symbol_system = Base37SymbolSystem()
    
    # Configure custom phase
    symbol_system.phase_config = {
        'input_format': 'binary',
        'processing_mode': 'advanced',
        'phase': 'phasei'
    }
    
    # Process with custom configuration
    data = b"Advanced Phase Processing"
    result = symbol_system.process_binary_phase(data)
    
    print(f"Input Data: {data}")
    print(f"Phase Config: {symbol_system.phase_config}")
    print("\nProcessing Results:")
    print(f"Initial Phase: {result['initial']['binary'][:32]}...")
    print(f"Block Count: {len(result['initial']['blocks'])}")
    print(f"Transform Phase: {result['transform']['transformed'][:32]}...")
    print(f"Segment Count: {len(result['transform']['segments'])}")
    print(f"Final Phase: {result['final']['processed'][:32]}...")
    print(f"Verification: {result['final']['verification']}")

def batch_processing_example():
    """Batch pattern processing example"""
    print("\n=== Batch Processing ===")
    
    engine = PatternEngine()
    patterns = [
        "HELLO", "WORLD", "START", "SYSTEM",
        "TEST123", "PATTERN", "BASE37", "COMPLEX"
    ]
    
    print(f"Processing {len(patterns)} patterns...")
    start_time = datetime.now()
    
    results = []
    for pattern in patterns:
        try:
            result = engine.process_pattern(pattern, secure=True)
            results.append({
                'pattern': pattern,
                'encoded': result['encoding']['value'],
                'checksum': result['checksum'],
                'success': True
            })
        except Exception as e:
            results.append({
                'pattern': pattern,
                'error': str(e),
                'success': False
            })
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    print("\nProcessing Results:")
    print(f"Total Time: {processing_time:.3f} seconds")
    print(f"Average Time: {(processing_time/len(patterns))*1000:.2f}ms per pattern")
    
    print("\nPattern Results:")
    for result in results:
        if result['success']:
            print(f"{result['pattern']}: {result['encoded']} (checksum: {result['checksum']})")
        else:
            print(f"{result['pattern']}: Error - {result['error']}")

def error_handling_example():
    """Advanced error handling example"""
    print("\n=== Advanced Error Handling ===")
    
    engine = PatternEngine()
    validator = ValidationFramework()
    
    test_patterns = [
        "VALID123",      # Valid pattern
        "INVALID@123",   # Invalid characters
        "TOO_LONG_PATTERN",  # Exceeds length
        "",             # Empty pattern
        "12345678900"   # Exceeds value range
    ]
    
    print("Testing error handling...")
    for pattern in test_patterns:
        print(f"\nTesting pattern: '{pattern}'")
        
        try:
            # Validate first
            validation = validator.validate(pattern)
            if not validation['valid']:
                print(f"Validation failed: {validation['errors']}")
                continue
                
            # Process if valid
            result = engine.process_pattern(pattern)
            print(f"Success: {result['encoding']['value']}")
            
        except ValueError as ve:
            print(f"Value Error: {str(ve)}")
        except Exception as e:
            print(f"General Error: {str(e)}")

def security_analysis_example():
    """Security analysis example"""
    print("\n=== Security Analysis ===")
    
    engine = PatternEngine()
    pattern = "SECURE123"
    
    # Process with security
    result = engine.process_pattern(pattern, secure=True)
    
    print(f"Input Pattern: {pattern}")
    print("\nSecurity Analysis:")
    print(f"Private Key: {result['security']['private_key']}")
    print(f"Public Key: {result['security']['public_key']}")
    print(f"Timestamp: {result['security']['timestamp']}")
    
    # Analyze transformations
    transform = result['transformation']
    print("\nTransformation Analysis:")
    print(f"Binary Length: {len(transform['binary'])} bits")
    print(f"Matrix Size: {len(transform['matrix'])}x{len(transform['matrix'][0])}")
    print(f"XOR Length: {len(transform['xored'])} bits")
    print(f"Final Length: {len(transform['final'])} bits")
    
    # Calculate entropy
    binary_data = transform['binary']
    entropy = -sum((binary_data.count(bit)/len(binary_data)) * 
                  np.log2(binary_data.count(bit)/len(binary_data))
                  for bit in set(binary_data))
    
    print(f"\nEntropy Analysis:")
    print(f"Binary Entropy: {entropy:.2f} bits")
    print(f"Theoretical Max: {np.log2(len(binary_data)):.2f} bits")

def main():
    """Run all advanced examples"""
    print("=== Base-37 Pattern Processing System - Advanced Examples ===")
    
    matrix_transformation_example()
    custom_phase_example()
    batch_processing_example()
    error_handling_example()
    security_analysis_example()
    
    print("\nAdvanced examples completed!")

if __name__ == "__main__":
    main() 