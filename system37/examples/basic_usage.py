#!/usr/bin/env python3

from system37 import PatternEngine, Base37SymbolSystem, ValidationFramework

def basic_example():
    """Basic pattern processing example"""
    print("\n=== Basic Pattern Processing ===")
    
    # Initialize engine
    engine = PatternEngine()
    
    # Process simple pattern
    pattern = "HELLO"
    result = engine.process_pattern(pattern)
    
    print(f"Input Pattern: {pattern}")
    print(f"Encoded: {result['encoding']['value']}")  # #258741%
    print(f"Checksum: {result['checksum']}")
    print(f"Structure: {result['structure']}")

def secure_example():
    """Secure pattern processing example"""
    print("\n=== Secure Pattern Processing ===")
    
    engine = PatternEngine()
    
    # Process with security features
    pattern = "WORLD"
    result = engine.process_pattern(pattern, secure=True)
    
    print(f"Input Pattern: {pattern}")
    print("\nSecurity Data:")
    print(f"Private Key: {result['security']['private_key']}")
    print(f"Public Key: {result['security']['public_key']}")
    
    print("\nTransformation Data:")
    print(f"Binary: {result['transformation']['binary']}")
    print(f"Base64: {result['transformation']['base64']}")
    print(f"Final: {result['transformation']['final']}")

def binary_phase_example():
    """Binary phase processing example"""
    print("\n=== Binary Phase Processing ===")
    
    symbol_system = Base37SymbolSystem()
    
    # Process binary data
    data = b"Hello World"
    result = symbol_system.process_binary_phase(data)
    
    print(f"Input Data: {data}")
    print("\nPhase Results:")
    print(f"Initial Phase: {result['initial']['binary'][:32]}...")
    print(f"Transform Phase: {result['transform']['transformed'][:32]}...")
    print(f"Final Phase: {result['final']['processed'][:32]}...")
    print(f"Checksum: {result['final']['checksum']}")
    print(f"Verified: {result['final']['verification']}")

def validation_example():
    """Pattern validation example"""
    print("\n=== Pattern Validation ===")
    
    validator = ValidationFramework()
    
    # Valid pattern
    valid_pattern = "#258741%"
    valid_result = validator.validate(valid_pattern)
    
    print(f"Valid Pattern: {valid_pattern}")
    print(f"Validation Result: {valid_result['valid']}")
    
    # Invalid pattern
    invalid_pattern = "#ABC@123%"
    invalid_result = validator.validate(invalid_pattern)
    
    print(f"\nInvalid Pattern: {invalid_pattern}")
    print(f"Validation Result: {invalid_result['valid']}")
    print(f"Errors: {invalid_result['errors']}")

def known_patterns_example():
    """Known patterns example"""
    print("\n=== Known Patterns ===")
    
    symbol_system = Base37SymbolSystem()
    
    # Show all known patterns
    print("Known Patterns:")
    for text, pattern in symbol_system.known_patterns.items():
        print(f"{text}: {pattern}")
    
    # Encode/decode known pattern
    text = "HELLO"
    encoded = symbol_system.encode_text(text)
    decoded = symbol_system.decode_text(encoded)
    
    print(f"\nEncoding '{text}':")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

def performance_example():
    """Performance monitoring example"""
    print("\n=== Performance Monitoring ===")
    
    engine = PatternEngine()
    
    # Process multiple patterns
    patterns = ["HELLO", "WORLD", "START", "SYSTEM"]
    for pattern in patterns:
        engine.process_pattern(pattern)
    
    # Get performance metrics
    metrics = engine.get_performance_metrics()
    
    print("Performance Metrics:")
    print(f"Average Response Time: {metrics['response_time']['average']:.2f}ms")
    print(f"Min Response Time: {metrics['response_time']['min']:.2f}ms")
    print(f"Max Response Time: {metrics['response_time']['max']:.2f}ms")
    print(f"Throughput: {metrics['throughput']} patterns/sec")
    print(f"Accuracy: {metrics['accuracy']*100:.3f}%")

def main():
    """Run all examples"""
    print("=== Base-37 Pattern Processing System Examples ===")
    
    basic_example()
    secure_example()
    binary_phase_example()
    validation_example()
    known_patterns_example()
    performance_example()
    
    print("\nExamples completed!")

if __name__ == "__main__":
    main() 