# Mixed Code Samples Dataset
# 混合代码样本 - 包含人类和AI风格的混合代码

import os
import json
from typing import List, Dict, Optional

# Human style - simple utility functions
def get_file_size(path):
    return os.path.getsize(path)

def read_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# AI style - comprehensive data validation class
class DataValidator:
    """
    Comprehensive data validation utility with advanced validation rules.
    
    This class provides sophisticated validation mechanisms for various
    data types with detailed error reporting and logging capabilities.
    """
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize the validator with configuration parameters.
        
        Args:
            config: Configuration dictionary containing validation rules
        """
        self.config = config
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def validate_email(self, email: str) -> bool:
        """
        Validate email address using comprehensive validation rules.
        
        Args:
            email: Email address string to validate
            
        Returns:
            True if email is valid, False otherwise
        """
        if not email or '@' not in email:
            self.validation_errors.append("Invalid email format")
            return False
        
        # Additional sophisticated validation logic
        local_part, domain = email.split('@', 1)
        if len(local_part) < 1 or len(domain) < 3:
            self.validation_errors.append("Email components too short")
            return False
        
        return True

# Human style - quick and dirty data processing
users = [
    {'name': 'Alice', 'email': 'alice@example.com'},
    {'name': 'Bob', 'email': 'bob@test.com'}
]

valid_users = []
for user in users:
    if '@' in user['email']:
        valid_users.append(user)

# AI style - comprehensive file processing system
class FileProcessor:
    """
    Advanced file processing system with comprehensive error handling.
    
    This class implements sophisticated file processing algorithms with
    support for multiple file formats, batch processing, and detailed
    progress tracking capabilities.
    """
    
    def __init__(self, processing_config: Optional[Dict[str, any]] = None):
        """
        Initialize file processor with optional configuration.
        
        Args:
            processing_config: Optional configuration for processing parameters
        """
        self.config = processing_config or self._get_default_config()
        self.processed_files: List[str] = []
        self.failed_files: List[str] = []
        self.processing_statistics = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_size_processed': 0
        }
    
    def _get_default_config(self) -> Dict[str, any]:
        """
        Retrieve default configuration parameters for file processing.
        
        Returns:
            Dictionary containing default configuration values
        """
        return {
            'max_file_size': 1024 * 1024 * 10,  # 10MB
            'supported_extensions': ['.txt', '.json', '.csv', '.log'],
            'batch_size': 100,
            'enable_compression': True,
            'backup_processed_files': False
        }
    
    def process_files(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Process multiple files with comprehensive error handling and statistics.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary containing processing results and statistics
        """
        self.processing_statistics['total_files'] = len(file_paths)
        
        for file_path in file_paths:
            try:
                if self._validate_file(file_path):
                    self._process_single_file(file_path)
                    self.processed_files.append(file_path)
                    self.processing_statistics['successful_files'] += 1
                else:
                    self.failed_files.append(file_path)
                    self.processing_statistics['failed_files'] += 1
            except Exception as e:
                self.failed_files.append(file_path)
                self.processing_statistics['failed_files'] += 1
                print(f"Error processing {file_path}: {str(e)}")
        
        return {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'statistics': self.processing_statistics
        }

# Human style - simple helper functions
def count_lines(filename):
    try:
        with open(filename, 'r') as f:
            return len(f.readlines())
    except:
        return 0

def backup_file(src, dst):
    import shutil
    shutil.copy2(src, dst)

# AI style - comprehensive logging and monitoring system
import logging
from datetime import datetime

class AdvancedLogger:
    """
    Advanced logging system with comprehensive monitoring capabilities.
    
    This class provides sophisticated logging mechanisms with support for
    multiple output formats, log rotation, and advanced filtering options.
    """
    
    def __init__(self, logger_name: str, log_level: str = 'INFO'):
        """
        Initialize advanced logger with comprehensive configuration.
        
        Args:
            logger_name: Name identifier for the logger instance
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Configure comprehensive logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Setup file handler with rotation
        file_handler = logging.FileHandler(f'{logger_name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_with_context(self, level: str, message: str, context: Optional[Dict[str, any]] = None):
        """
        Log message with additional contextual information.
        
        Args:
            level: Logging level for the message
            message: Primary log message
            context: Optional dictionary containing additional context
        """
        if context:
            enhanced_message = f"{message} | Context: {json.dumps(context)}"
        else:
            enhanced_message = message
        
        getattr(self.logger, level.lower())(enhanced_message)

# Human style - main execution
if __name__ == '__main__':
    # Simple test
    files = ['test1.txt', 'test2.txt']
    processor = FileProcessor()
    results = processor.process_files(files)
    print(f"Processed: {len(results['processed_files'])}")
    
    # Quick validation
    validator = DataValidator({})
    is_valid = validator.validate_email('test@example.com')
    print(f"Email valid: {is_valid}")

# AI style - comprehensive demonstration and testing
def demonstrate_comprehensive_functionality():
    """
    Demonstrate the complete functionality of all components with detailed examples.
    
    This function provides a comprehensive demonstration of all available
    features including error handling, logging, and performance monitoring.
    """
    # Initialize comprehensive logging system
    logger = AdvancedLogger('demonstration_logger', 'DEBUG')
    logger.log_with_context('info', 'Starting comprehensive functionality demonstration')
    
    try:
        # Demonstrate file processing capabilities
        processor = FileProcessor({
            'max_file_size': 1024 * 1024 * 5,  # 5MB
            'batch_size': 50,
            'enable_compression': True
        })
        
        # Create sample file list for processing
        sample_files = [f'sample_file_{i}.txt' for i in range(10)]
        
        logger.log_with_context('info', 'Processing sample files', {
            'file_count': len(sample_files),
            'processor_config': processor.config
        })
        
        # Execute file processing with comprehensive monitoring
        processing_results = processor.process_files(sample_files)
        
        # Log detailed processing results
        logger.log_with_context('info', 'File processing completed', {
            'results': processing_results['statistics']
        })
        
        # Demonstrate data validation capabilities
        validator = DataValidator({'strict_mode': True, 'enable_warnings': True})
        
        test_emails = [
            'valid@example.com',
            'invalid-email',
            'another@test.org'
        ]
        
        for email in test_emails:
            validation_result = validator.validate_email(email)
            logger.log_with_context('debug', f'Email validation result: {validation_result}', {
                'email': email,
                'errors': validator.validation_errors,
                'warnings': validator.validation_warnings
            })
        
        logger.log_with_context('info', 'Comprehensive demonstration completed successfully')
        
    except Exception as e:
        logger.log_with_context('error', f'Demonstration failed: {str(e)}', {
            'error_type': type(e).__name__,
            'error_details': str(e)
        })
        raise

# Execute comprehensive demonstration if run as main module
if __name__ == '__main__' and len(os.sys.argv) > 1 and os.sys.argv[1] == '--comprehensive':
    demonstrate_comprehensive_functionality() 