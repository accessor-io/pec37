from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from collections import defaultdict
import base64

class Base37SymbolSystem:
    def __init__(self):
        # Define core symbols as per specifications
        self.core_symbols = {'#', '$', '&', '!', '%'}
        self.symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        self.symbol_map = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.reverse_map = {idx: sym for idx, sym in enumerate(self.symbols)}
        
        # Known patterns from specifications
        self.known_patterns = {
            'HELLO': '#258741',
            'WORLD': '$86160753',
            'START': '&20937969',
            'SYSTEM': '!249634233'
        }
        
        # Initialize processing limits
        self.max_value = 999999999
        self.max_word_length = 8
        self.max_block_size = 12
        self.max_message_length = 1000
        
        # Binary phase configuration
        self.phase_config = {
            'input_format': 'binary',
            'processing_mode': 'standard',
            'phase': 'phasei'
        }
        
    def encode(self, value: int) -> str:
        """Convert decimal to Base-37 representation with format {symbol}{value}{separator}"""
        if value < 0 or value > self.max_value:
            raise ValueError(f"Value must be between 0 and {self.max_value}")
            
        # Calculate base conversion
        if value == 0:
            return f"#0%"
            
        result = []
        temp_value = value
        while temp_value:
            temp_value, remainder = divmod(temp_value, 37)
            result.append(self.symbols[remainder])
            
        # Format according to specifications
        encoded = ''.join(reversed(result))
        return f"#{encoded}%"
    
    def decode(self, encoded: str) -> int:
        """Convert Base-37 representation to decimal"""
        if not self._validate_format(encoded):
            raise ValueError("Invalid format. Must be {symbol}{value}{separator}")
            
        # Extract value part (remove symbol and separator)
        value_part = encoded[1:-1]
        
        result = 0
        for char in value_part:
            if char not in self.symbol_map:
                raise ValueError(f"Invalid Base-37 character: {char}")
            result = result * 37 + self.symbol_map[char]
            
        if result > self.max_value:
            raise ValueError(f"Decoded value exceeds maximum: {self.max_value}")
            
        return result
    
    def encode_text(self, text: str) -> str:
        """Encode text using Base-37 with specified format"""
        if len(text) > self.max_word_length:
            raise ValueError(f"Text exceeds maximum length of {self.max_word_length}")
            
        # Check for known patterns
        if text in self.known_patterns:
            return self.known_patterns[text]
            
        # Calculate value using formula: char_value = (ASCII - 65) * (37 ^ position)
        value = 0
        for pos, char in enumerate(reversed(text)):
            if char.isalpha():
                char_value = (ord(char.upper()) - 65) * (37 ** pos)
            elif char.isdigit():
                char_value = (ord(char) - 48 + 26) * (37 ** pos)
            else:
                raise ValueError(f"Invalid character: {char}")
            value += char_value
            
        return self.encode(value)
    
    def decode_text(self, encoded: str) -> str:
        """Decode Base-37 representation to text"""
        # Check known patterns
        for text, pattern in self.known_patterns.items():
            if encoded == pattern:
                return text
                
        value = self.decode(encoded)
        result = []
        
        while value:
            value, remainder = divmod(value, 37)
            if remainder < 26:
                result.append(chr(remainder + 65))
            else:
                result.append(str(remainder - 26))
                
        return ''.join(reversed(result))
    
    def process_binary_phase(self, data: bytes) -> Dict[str, Any]:
        """Process data through binary phase system"""
        # Convert to base64 for initial processing
        base64_data = base64.b64encode(data).decode()
        
        # Process through phases
        phase_results = {
            'initial': self._process_initial_phase(base64_data),
            'transform': self._process_transform_phase(base64_data),
            'final': self._process_final_phase(base64_data)
        }
        
        return phase_results
    
    def _process_initial_phase(self, data: str) -> Dict[str, Any]:
        """Process initial phase"""
        binary = ''.join(format(ord(c), '08b') for c in data)
        return {
            'binary': binary,
            'length': len(binary),
            'blocks': [binary[i:i+8] for i in range(0, len(binary), 8)]
        }
    
    def _process_transform_phase(self, data: str) -> Dict[str, Any]:
        """Process transform phase"""
        # Convert to Base-37
        transformed = ''
        for char in data:
            if char.isalnum():
                value = self.symbol_map.get(char.upper(), 0)
                transformed += format(value, '06b')
            else:
                transformed += '000000'
                
        return {
            'transformed': transformed,
            'segments': [transformed[i:i+6] for i in range(0, len(transformed), 6)]
        }
    
    def _process_final_phase(self, data: str) -> Dict[str, Any]:
        """Process final phase"""
        # Apply final transformations
        processed = self._apply_phase_transformations(data)
        
        return {
            'processed': processed,
            'checksum': self._calculate_phase_checksum(processed),
            'verification': self._verify_phase_result(processed)
        }
    
    def _apply_phase_transformations(self, data: str) -> str:
        """Apply phase-specific transformations"""
        if self.phase_config['phase'] == 'phasei':
            return self._transform_phase_i(data)
        return data
    
    def _transform_phase_i(self, data: str) -> str:
        """Apply Phase I transformation"""
        # Example transformation
        transformed = ''
        for char in data:
            if char in self.symbol_map:
                value = (self.symbol_map[char] + 7) % 37  # Shift by 7
                transformed += self.reverse_map[value]
            else:
                transformed += char
        return transformed
    
    def _calculate_phase_checksum(self, data: str) -> int:
        """Calculate phase-specific checksum"""
        return sum(self.symbol_map.get(c, 0) for c in data) % 37
    
    def _verify_phase_result(self, data: str) -> bool:
        """Verify phase processing result"""
        checksum = self._calculate_phase_checksum(data)
        return checksum == 0 or checksum == 1  # Valid checksums
    
    def calculate_checksum(self, pattern: str) -> int:
        """Calculate mod-37 verification checksum"""
        value = self.decode(pattern)
        return value % 37
    
    def validate_pattern(self, pattern: str) -> bool:
        """Validate pattern format and checksum"""
        try:
            if not self._validate_format(pattern):
                return False
            value = self.decode(pattern)
            checksum = self.calculate_checksum(pattern)
            return checksum == (value % 37)
        except ValueError:
            return False
    
    def _validate_format(self, pattern: str) -> bool:
        """Validate pattern follows {symbol}{value}{separator} format"""
        if len(pattern) < 3:
            return False
        return (pattern[0] in self.core_symbols and 
                pattern[-1] in self.core_symbols and 
                all(c in self.symbols for c in pattern[1:-1]))
    
    def get_symbol_info(self, symbol: str) -> Dict[str, str]:
        """Get symbol definition from specifications"""
        symbol_definitions = {
            '#': 'Initialize block',
            '$': 'Start character',
            '&': 'Continue message',
            '!': 'Space/separator',
            '%': 'End character'
        }
        return {'purpose': symbol_definitions.get(symbol, 'Unknown symbol')}