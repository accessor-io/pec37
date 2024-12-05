#!/usr/bin/env python3

import argparse
from system37 import PatternEngine, Base37SymbolSystem, ValidationFramework

def process_command(command: str, secure: bool = False):
    """Process a single command"""
    engine = PatternEngine()
    
    try:
        result = engine.process_pattern(command, secure=secure)
        
        print("\n=== Command Processing Result ===")
        print(f"Input Command: {command}")
        print(f"Encoded Value: {result['encoding']['value']}")
        print(f"Checksum: {result['checksum']}")
        
        if secure:
            print("\nSecurity Data:")
            print(f"Private Key: {result['security']['private_key']}")
            print(f"Public Key: {result['security']['public_key']}")
            print("\nTransformation Data:")
            print(f"Binary: {result['transformation']['binary'][:32]}...")
            print(f"Final: {result['transformation']['final'][:32]}...")
            
        return True
        
    except Exception as e:
        print(f"\nError processing command: {str(e)}")
        return False

def interactive_mode():
    """Run in interactive mode"""
    engine = PatternEngine()
    symbol_system = Base37SymbolSystem()
    
    print("\n=== Base-37 Pattern Processor Interactive Mode ===")
    print("Type 'exit' to quit, 'help' for commands")
    
    while True:
        try:
            command = input("\nEnter command > ").strip().upper()
            
            if command == 'EXIT':
                break
            elif command == 'HELP':
                print("\nAvailable Commands:")
                print("  <pattern>  - Process a pattern (e.g., HELLO)")
                print("  -s <pat>   - Process with security (e.g., -s HELLO)")
                print("  known      - List known patterns")
                print("  help       - Show this help")
                print("  exit       - Exit the program")
                continue
            elif command == 'KNOWN':
                print("\nKnown Patterns:")
                for text, pattern in symbol_system.known_patterns.items():
                    print(f"  {text}: {pattern}")
                continue
            elif command.startswith('-S '):
                # Secure processing
                pattern = command[3:]
                process_command(pattern, secure=True)
            else:
                # Normal processing
                process_command(command)
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Base-37 Pattern Processor')
    parser.add_argument('command', nargs='?', help='Command to process')
    parser.add_argument('-s', '--secure', action='store_true', help='Enable secure processing')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.command:
        process_command(args.command.upper(), args.secure)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 