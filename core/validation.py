from typing import Dict, List, Any, Callable
from datetime import datetime
import re

class ValidationFramework:
    def __init__(self):
        self.validation_rules = {}
        self.error_handlers = {}
        self.validation_history = []
        
    def add_validation_rule(self, rule_name: str, rule_func: Callable, error_handler: Callable = None):
        """Add a new validation rule with optional error handler"""
        self.validation_rules[rule_name] = rule_func
        if error_handler:
            self.error_handlers[rule_name] = error_handler
            
    def validate(self, data: Any, rules: List[str] = None) -> Dict[str, Any]:
        """Validate data against specified rules or all rules"""
        if rules is None:
            rules = list(self.validation_rules.keys())
            
        results = {
            'timestamp': datetime.now(),
            'valid': True,
            'rule_results': {},
            'errors': []
        }
        
        for rule_name in rules:
            if rule_name not in self.validation_rules:
                raise ValueError(f"Unknown validation rule: {rule_name}")
                
            try:
                rule_result = self.validation_rules[rule_name](data)
                results['rule_results'][rule_name] = rule_result
                if not rule_result:
                    results['valid'] = False
                    if rule_name in self.error_handlers:
                        error_info = self.error_handlers[rule_name](data)
                        results['errors'].append({
                            'rule': rule_name,
                            'error_info': error_info
                        })
            except Exception as e:
                results['valid'] = False
                results['errors'].append({
                    'rule': rule_name,
                    'error': str(e)
                })
                
        self.validation_history.append(results)
        return results
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get the history of validation results"""
        return self.validation_history
    
    def clear_validation_history(self):
        """Clear the validation history"""
        self.validation_history = []
        
    # Common validation rules
    @staticmethod
    def pattern_length_rule(pattern: str, min_length: int = 1, max_length: int = 100) -> bool:
        """Validate pattern length"""
        return min_length <= len(pattern) <= max_length
    
    @staticmethod
    def symbol_set_rule(pattern: str, valid_symbols: set) -> bool:
        """Validate pattern contains only valid symbols"""
        return all(char in valid_symbols for char in pattern)
    
    @staticmethod
    def pattern_format_rule(pattern: str, format_regex: str) -> bool:
        """Validate pattern format using regex"""
        return bool(re.match(format_regex, pattern))
    
    @staticmethod
    def sequence_rule(pattern: str, valid_sequences: List[str]) -> bool:
        """Validate pattern contains valid sequences"""
        return any(seq in pattern for seq in valid_sequences) 