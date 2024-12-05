from typing import Dict, List, Any, Callable, Optional, Set, Union
from datetime import datetime
import re
from collections import defaultdict
import numpy as np

class ValidationFramework:
    def __init__(self):
        self.validation_rules = {}
        self.error_handlers = {}
        self.validation_history = []
        self.rule_dependencies = defaultdict(set)
        self.rule_metrics = defaultdict(lambda: {'passes': 0, 'failures': 0, 'time': 0})
        self.validation_cache = {}
        
        # Initialize default rules based on specifications
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize validation rules from specifications"""
        # Pattern structure validation
        self.add_validation_rule(
            'format_check',
            lambda p: self.pattern_format_rule(p, r'^[#$&!%][0-9A-Z]+[#$&!%]$')
        )
        
        # Value range validation
        self.add_validation_rule(
            'value_range',
            lambda p: self.value_range_rule(p, 0, 999999999)
        )
        
        # Symbol set validation
        self.add_validation_rule(
            'symbol_set',
            lambda p: self.symbol_set_rule(p, {'#', '$', '&', '!', '%'})
        )
        
        # Checksum validation
        self.add_validation_rule(
            'checksum',
            lambda p: self.checksum_rule(p)
        )
        
        # Block size validation
        self.add_validation_rule(
            'block_size',
            lambda p: self.block_size_rule(p, max_size=12)
        )
        
        # Message length validation
        self.add_validation_rule(
            'message_length',
            lambda p: self.message_length_rule(p, max_length=1000)
        )
        
    def validate(self, 
                data: Any, 
                rules: Optional[List[str]] = None, 
                use_cache: bool = True) -> Dict[str, Any]:
        """Validate data against specified rules or all rules"""
        if rules is None:
            rules = list(self.validation_rules.keys())
            
        # Check cache if enabled
        cache_key = self._compute_cache_key(data, rules)
        if use_cache and cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
            
        start_time = datetime.now()
        results = {
            'timestamp': start_time,
            'valid': True,
            'rule_results': {},
            'errors': [],
            'execution_order': self._get_execution_order(rules)
        }
        
        # Execute rules in dependency order
        for rule_name in results['execution_order']:
            if rule_name not in rules:
                continue
                
            rule_start = datetime.now()
            try:
                rule_result = self.validation_rules[rule_name](data)
                results['rule_results'][rule_name] = rule_result
                
                # Update metrics
                execution_time = (datetime.now() - rule_start).total_seconds()
                self.rule_metrics[rule_name]['time'] += execution_time
                
                if rule_result:
                    self.rule_metrics[rule_name]['passes'] += 1
                else:
                    self.rule_metrics[rule_name]['failures'] += 1
                    results['valid'] = False
                    
                    if rule_name in self.error_handlers:
                        error_info = self.error_handlers[rule_name](data)
                        results['errors'].append({
                            'rule': rule_name,
                            'error_info': error_info,
                            'timestamp': datetime.now()
                        })
                        
            except Exception as e:
                results['valid'] = False
                results['errors'].append({
                    'rule': rule_name,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
                self.rule_metrics[rule_name]['failures'] += 1
                
        # Calculate metrics
        results['metrics'] = {
            'total_time': (datetime.now() - start_time).total_seconds(),
            'rules_executed': len(results['rule_results']),
            'error_count': len(results['errors']),
            'rule_metrics': self._get_rule_metrics(rules)
        }
        
        # Update cache and history
        if use_cache:
            self.validation_cache[cache_key] = results
        self.validation_history.append(results)
        
        return results
    
    @staticmethod
    def pattern_format_rule(pattern: str, format_regex: str) -> bool:
        """Validate pattern format using regex"""
        return bool(re.match(format_regex, pattern))
    
    @staticmethod
    def value_range_rule(pattern: str, min_value: int, max_value: int) -> bool:
        """Validate pattern value range"""
        try:
            value = int(pattern[1:-1])
            return min_value <= value <= max_value
        except ValueError:
            return False
    
    @staticmethod
    def symbol_set_rule(pattern: str, valid_symbols: Set[str]) -> bool:
        """Validate pattern contains only valid symbols"""
        return (pattern[0] in valid_symbols and 
                pattern[-1] in valid_symbols and 
                all(c.isalnum() for c in pattern[1:-1]))
    
    @staticmethod
    def checksum_rule(pattern: str) -> bool:
        """Validate pattern checksum (mod-37)"""
        try:
            value = int(pattern[1:-1])
            return value % 37 == 0
        except ValueError:
            return False
    
    @staticmethod
    def block_size_rule(pattern: str, max_size: int) -> bool:
        """Validate pattern block size"""
        return len(pattern) <= max_size
    
    @staticmethod
    def message_length_rule(pattern: str, max_length: int) -> bool:
        """Validate total message length"""
        return len(pattern) <= max_length
    
    def _compute_cache_key(self, data: Any, rules: List[str]) -> str:
        """Compute cache key for data and rules"""
        rules_str = ','.join(sorted(rules))
        return f"{hash(str(data))}:{hash(rules_str)}"
    
    def _get_execution_order(self, rules: List[str]) -> List[str]:
        """Determine rule execution order based on dependencies"""
        visited = set()
        order = []
        
        def visit(rule):
            if rule in visited:
                return
            visited.add(rule)
            for dep in self.rule_dependencies[rule]:
                if dep in self.validation_rules:
                    visit(dep)
            order.append(rule)
            
        for rule in rules:
            visit(rule)
            
        return order
    
    def _get_rule_metrics(self, rules: List[str]) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get metrics for specified rules"""
        return {
            rule: {
                'passes': self.rule_metrics[rule]['passes'],
                'failures': self.rule_metrics[rule]['failures'],
                'average_time': (
                    self.rule_metrics[rule]['time'] / 
                    (self.rule_metrics[rule]['passes'] + self.rule_metrics[rule]['failures'])
                    if self.rule_metrics[rule]['passes'] + self.rule_metrics[rule]['failures'] > 0
                    else 0
                )
            }
            for rule in rules
        }