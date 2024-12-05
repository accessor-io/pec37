from typing import Dict, List, Optional
import string

class Base37SymbolSystem:
    def __init__(self):
        # Define Base-37 symbol set (0-9, A-Z, and special character)
        self.symbols = string.digits + string.ascii_uppercase + '_'
        self.symbol_map = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.reverse_map = {idx: sym for idx, sym in enumerate(self.symbols)}
        
    def encode(self, value: int) -> str:
        """Convert decimal to Base-37 representation"""
        if value < 0:
            raise ValueError("Cannot encode negative values")
            
        if value == 0:
            return self.symbols[0]
            
        result = []
        while value:
            value, remainder = divmod(value, 37)
            result.append(self.symbols[remainder])
        return ''.join(reversed(result))
    
    def decode(self, encoded: str) -> int:
        """Convert Base-37 representation to decimal"""
        result = 0
        for char in encoded:
            if char not in self.symbol_map:
                raise ValueError(f"Invalid Base-37 character: {char}")
            result = result * 37 + self.symbol_map[char]
        return result
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is part of the Base-37 system"""
        return all(char in self.symbols for char in symbol)
    
    def get_symbol_value(self, symbol: str) -> Optional[int]:
        """Get the decimal value of a Base-37 symbol"""
        return self.symbol_map.get(symbol)
    
    def get_symbol(self, value: int) -> Optional[str]:
        """Get the symbol for a given decimal value"""
        return self.reverse_map.get(value)
    
    def get_all_symbols(self) -> List[str]:
        """Return all valid Base-37 symbols"""
        return list(self.symbols)
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed information about a symbol"""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
            
        return {
            'symbol': symbol,
            'value': self.get_symbol_value(symbol),
            'is_digit': symbol in string.digits,
            'is_letter': symbol in string.ascii_uppercase,
            'is_special': symbol == '_'
        } 