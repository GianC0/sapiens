"""
Production-ready AutoGen workflow for automated model implementation.
Generates deep learning models from research papers with full validation and monitoring.
"""

import ast
import base64
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import autogen
import pandas as pd
import PyPDF2
import requests
import yaml
from autogen.coding import LocalCommandLineCodeExecutor
from github import Github
from retry import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerationStatus(Enum):
    """Model generation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    VALIDATING = "validating"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class GenerationConfig:
    """Configuration for model generation"""
    max_iterations: int = 15
    timeout_seconds: float = 3600
    test_timeout: float = 300
    max_retries: int = 3
    retry_backoff: float = 2.0
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    save_conversations: bool = True
    enable_code_validation: bool = True
    enable_security_checks: bool = True
    max_code_size_mb: float = 10.0
    allowed_imports: List[str] = field(default_factory=lambda: [
        'torch', 'numpy', 'pandas', 'sklearn', 'models', 'typing',
        'pathlib', 'json', 'pickle', 'logging', 'mlflow', 'math'
    ])
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r'exec\s*\(', r'eval\s*\(', r'__import__', r'compile\s*\(',
        r'subprocess', r'os\.system', r'open\s*\(.+[\'"]w'
    ])
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 8000
    enable_parallel: bool = False
    max_parallel_jobs: int = 3


@dataclass
class GenerationMetrics:
    """Metrics for generation process"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: GenerationStatus = GenerationStatus.PENDING
    iterations: int = 0
    test_runs: int = 0
    errors: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    code_size_bytes: int = 0
    conversation_size_bytes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'iterations': self.iterations,
            'test_runs': self.test_runs,
            'errors': self.errors[-10:],  # Keep last 10 errors
            'validation_results': self.validation_results,
            'memory_usage_mb': self.memory_usage_mb,
            'code_size_bytes': self.code_size_bytes
        }


class CodeValidator:
    """Validate generated code for security and compliance"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    def validate_code(self, code: str, filename: str = "generated.py") -> Tuple[bool, List[str]]:
        """Validate code for security and syntax issues"""
        errors = []
        
        # Check size
        code_size_mb = len(code.encode()) / 1024 / 1024
        if code_size_mb > self.config.max_code_size_mb:
            errors.append(f"Code too large: {code_size_mb:.1f}MB > {self.config.max_code_size_mb}MB")
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        
        # Security checks
        if self.config.enable_security_checks:
            for pattern in self.config.forbidden_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    errors.append(f"Forbidden pattern found: {pattern}")
            
            # Check imports
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in self.config.allowed_imports:
                            errors.append(f"Unauthorized import: {module}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in self.config.allowed_imports:
                            errors.append(f"Unauthorized import from: {module}")
        
        return len(errors) == 0, errors
    
    def sanitize_code(self, code: str) -> str:
        """Basic code sanitization"""
        # Remove potential security issues
        code = re.sub(r'#.*?API[_\s]?KEY.*', '# [REDACTED]', code, flags=re.IGNORECASE)
        code = re.sub(r'#.*?SECRET.*', '# [REDACTED]', code, flags=re.IGNORECASE)
        code = re.sub(r'#.*?PASSWORD.*', '# [REDACTED]', code, flags=re.IGNORECASE)
        return code


class DocumentProcessor:
    """Process documents (PDFs, GitHub repos) for context"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_pdf_text(self, pdf_path: str, max_pages: int = 50) -> str:
        """Extract text from PDF with caching"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return ""
        
        # Check cache
        cache_key = hashlib.md5(str(pdf_path).encode()).hexdigest()
        cache_file = self.cache_dir / f"pdf_{cache_key}.txt"
        
        if cache_file.exists():
            logger.info(f"Using cached PDF text for {pdf_path.name}")
            return cache_file.read_text()
        
        # Extract text
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = min(len(reader.pages), max_pages)
                
                for i in range(num_pages):
                    try:
                        page = reader.pages[i]
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {i}: {e}")
            
            # Cache result
            cache_file.write_text(text)
            logger.info(f"Extracted {len(text)} chars from {num_pages} pages")
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            return ""
    
    @retry(tries=3, delay=2, backoff=2)
    def fetch_github_code(self, github_url: str, files_pattern: str = "*.py") -> Dict[str, str]:
        """Fetch code from GitHub repository"""
        try:
            # Parse GitHub URL
            pattern = r'github\.com/([^/]+)/([^/]+)'
            match = re.search(pattern, github_url)
            if not match:
                logger.error(f"Invalid GitHub URL: {github_url}")
                return {}
            
            owner, repo = match.groups()
            
            # Use GitHub API if token available
            if os.environ.get('GITHUB_TOKEN'):
                g = Github(os.environ['GITHUB_TOKEN'])
                repo_obj = g.get_repo(f"{owner}/{repo}")
                
                files = {}
                contents = repo_obj.get_contents("")
                
                while contents:
                    file_content = contents.pop(0)
                    if file_content.type == "dir":
                        contents.extend(repo_obj.get_contents(file_content.path))
                    elif file_content.name.endswith('.py'):
                        try:
                            decoded = base64.b64decode(file_content.content).decode('utf-8')
                            files[file_content.path] = decoded
                        except Exception as e:
                            logger.warning(f"Failed to decode {file_content.path}: {e}")
                
                logger.info(f"Fetched {len(files)} Python files from GitHub")
                return files
            
            # Fallback to raw URLs
            logger.warning("No GitHub token, using limited raw access")
            raw_url = github_url.replace('github.com', 'raw.githubusercontent.com') + '/main'
            
            # Try common model files
            files = {}
            for filename in ['model.py', 'main.py', 'network.py', 'train.py']:
                try:
                    response = requests.get(f"{raw_url}/{filename}", timeout=10)
                    if response.status_code == 200:
                        files[filename] = response.text
                except:
                    pass
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to fetch GitHub code: {e}")
            return {}


class ModelGeneratorWorkflow:
    """Production-ready model generation workflow"""
    
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        github_url: Optional[str] = None,
        model_name: str = "GeneratedModel",
        config: Optional[GenerationConfig] = None
    ):
        self.pdf_path = pdf_path
        self.github_url = github_url
        self.model_name = self._sanitize_model_name(model_name)
        self.config = config or GenerationConfig()
        self.metrics = GenerationMetrics()
        
        # Setup directories
        self.model_dir = Path(f"models/{self.model_name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.workspace_dir = self.config.temp_dir / self.model_name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = CodeValidator(self.config)
        self.processor = DocumentProcessor(self.config.cache_dir)
        
        # Load context
        self._load_context()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Initialize agents (lazy)
        self._agents_initialized = False
    
    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name for filesystem"""
        return re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    def _load_context(self):
        """Load and process context documents"""
        logger.info(f"Loading context for {self.model_name}")
        
        # Load paper
        self.paper_content = ""
        if self.pdf_path:
            self.paper_content = self.processor.extract_pdf_text(self.pdf_path)
            logger.info(f"Loaded {len(self.paper_content)} chars from paper")
        
        # Load GitHub code
        self.github_code = {}
        if self.github_url:
            self.github_code = self.processor.fetch_github_code(self.github_url)
            logger.info(f"Loaded {len(self.github_code)} files from GitHub")
        
        # Load interface and tests
        self.interface_code = self._load_file("models/interfaces.py")
        self.test_code = self._load_file("tests/test_model_interface.py")
        self.utils_code = self._load_file("models/utils.py")
    
    def _load_file(self, path: str) -> str:
        """Load file with error handling"""
        try:
            return Path(path).read_text()
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return ""
    
    def _setup_monitoring(self):
        """Setup monitoring and health checks"""
        self.health_check_file = self.workspace_dir / "health.json"
        self._update_health_status("initialized")
    
    def _update_health_status(self, status: str, details: Dict = None):
        """Update health check file"""
        health_data = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'pid': os.getpid(),
            'details': details or {}
        }
        try:
            self.health_check_file.write_text(json.dumps(health_data, indent=2))
        except:
            pass
    
    def _setup_agents(self):
        """Initialize AutoGen agents"""
        if self._agents_initialized:
            return
        
        logger.info("Setting up AutoGen agents")
        
        # Configure LLM
        llm_config = {
            "config_list": [{
                "model": self.config.llm_model,
                "api_key": self._get_api_key(),
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }],
            "timeout": 120,
            "cache_seed": 42
        }
        
        # Code executor with timeout
        self.executor = LocalCommandLineCodeExecutor(
            timeout=int(self.config.test_timeout),
            work_dir=str(self.workspace_dir),
            virtual_env_context=None
        )
        
        # Developer agent
        self.developer = autogen.AssistantAgent(
            name="ModelDeveloper",
            llm_config=llm_config,
            system_message=self._build_developer_prompt(),
            max_consecutive_auto_reply=self.config.max_iterations
        )
        
        # Test runner agent
        self.tester = autogen.UserProxyAgent(
            name="TestRunner",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.config.max_iterations,
            code_execution_config={"executor": self.executor},
            is_termination_msg=self._check_termination,
            default_auto_reply="Continue fixing the errors."
        )
        
        # Code reviewer agent
        self.reviewer = autogen.AssistantAgent(
            name="CodeReviewer",
            llm_config=llm_config,
            system_message=self._build_reviewer_prompt()
        )
        
        self._agents_initialized = True
    
    def _get_api_key(self) -> str:
        """Get API key based on provider"""
        if self.config.llm_provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
        elif self.config.llm_provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key = os.environ.get("LLM_API_KEY")
        
        if not key:
            raise ValueError(f"API key not found for {self.config.llm_provider}")
        return key
    
    def _check_termination(self, msg: Dict) -> bool:
        """Check if generation should terminate"""
        content = msg.get("content", "").upper()
        return any(term in content for term in [
            "ALL TESTS PASSED",
            "GENERATION COMPLETE",
            "SUCCESS"
        ])
    
    def _build_developer_prompt(self) -> str:
        """Build comprehensive developer prompt"""
        
        # Truncate context to fit token limits
        max_paper_chars = 15000
        max_github_chars = 10000
        max_test_chars = 8000
        
        paper_excerpt = self.paper_content[:max_paper_chars] if self.paper_content else "No paper provided"
        
        github_excerpt = ""
        if self.github_code:
            for path, code in list(self.github_code.items())[:3]:  # First 3 files
                github_excerpt += f"\n=== {path} ===\n{code[:3000]}\n"
        
        test_excerpt = self.test_code[:max_test_chars]
        
        return f"""You are an expert PyTorch developer implementing financial models.

TASK: Implement '{self.model_name}' following the MarketModel interface.

=== INTERFACE (MUST IMPLEMENT) ===
{self.interface_code}

=== KEY TEST REQUIREMENTS ===
{test_excerpt}

=== RESEARCH PAPER ===
{paper_excerpt}

=== REFERENCE CODE ===
{github_excerpt or "No reference code"}

=== IMPLEMENTATION GUIDELINES ===

1. FILE STRUCTURE:
   - Main file: models/{self.model_name}/{self.model_name}.py
   - Create __init__.py importing the main class
   - Follow the exact interface from MarketModel

2. REQUIRED METHODS:
   - __init__: Initialize with exact parameters from interface
   - initialize(data): Train model, return validation loss (float)
   - update(data, current_time, active_mask): Warm-start training
   - predict(data, current_time, active_mask): Return Dict[ticker, float]
   - state_dict/load_state_dict: Model persistence

3. DATA HANDLING:
   - Input: Dict[ticker, DataFrame] with columns: Open, High, Low, Close/Adj Close, Volume
   - Use build_input_tensor/build_pred_tensor from models.utils
   - Handle active_mask for inactive instruments (set predictions to 0)
   - Maintain consistent universe ordering

4. ARCHITECTURE:
   - Follow paper architecture if provided
   - Default to 2-layer LSTM with dropout=0.2 if no paper
   - Input shape: (batch, sequence_len, n_stocks * features)
   - Output: scalar prediction per stock

5. TRAINING:
   - Split data using self.train_end timestamp
   - Use DataLoader with self.batch_size
   - Implement early stopping with self.patience
   - Save weights: init.pt (first training), latest.pt (updates)

6. MLFLOW:
   - Call mlflow.set_experiment() in _setup_mlflow()
   - Log metrics: best_validation_loss, train_loss, val_loss
   - Log hyperparameters

7. ERROR HANDLING:
   - Handle empty data gracefully
   - Validate tensor shapes
   - Check for NaN/Inf in predictions

Start with basic structure, then iteratively fix test failures.
After each implementation, the tests will run automatically.
"""
    
    def _build_reviewer_prompt(self) -> str:
        """Build code reviewer prompt"""
        return """You are a senior code reviewer for financial ML models.

Review the generated code for:
1. Interface compliance - all required methods implemented correctly
2. PyTorch best practices - proper tensor operations, device handling
3. Data handling - correct shape transformations, NaN handling
4. Training logic - proper train/valid split, loss calculation
5. Memory efficiency - no leaks, proper cleanup
6. Error handling - graceful failures, informative errors
7. Code quality - clear variable names, comments for complex logic

If you find issues, provide specific fixes with code examples.
Focus on correctness and robustness over optimization.
"""
    
    def _create_test_script(self) -> Path:
        """Create test runner script"""
        test_script = f"""#!/usr/bin/env python3
import sys
import os
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Run tests
cmd = [
    sys.executable, "-m", "pytest",
    "tests/test_model_interface.py::TestModelInterfaceProduction",
    "-xvs",
    "--tb=short",
    "-k", "not stress and not memory_leak",
    "--model_class", "models.{self.model_name}.{self.model_name}.{self.model_name}",
    "--timeout", "60"
]

result = subprocess.run(cmd, capture_output=True, text=True)

# Parse results
output = result.stdout + result.stderr
print(output)

# Check for success
if result.returncode == 0:
    print("\\n✅ ALL TESTS PASSED!")
    sys.exit(0)
else:
    # Extract failure info
    print("\\n❌ Tests failed")
    
    # Try to extract specific errors
    error_lines = []
    for line in output.split('\\n'):
        if 'FAILED' in line or 'ERROR' in line or 'assert' in line:
            error_lines.append(line)
    
    if error_lines:
        print("\\nKey errors:")
        for line in error_lines[:10]:  # First 10 errors
            print(f"  - {{line}}")
    
    sys.exit(1)
"""
        
        script_path = self.workspace_dir / "run_tests.py"
        script_path.write_text(test_script)
        script_path.chmod(0o755)
        return script_path
    
    @retry(tries=3, delay=5, backoff=2)
    def generate_model(self) -> bool:
        """Generate model with retries and monitoring"""
        logger.info(f"Starting generation for {self.model_name}")
        
        self.metrics.status = GenerationStatus.IN_PROGRESS
        self._update_health_status("generating")
        
        try:
            # Setup agents
            self._setup_agents()
            
            # Create test script
            test_script = self._create_test_script()
            
            # Save initial state
            self._save_checkpoint("initial")
            
            # Build initial message
            initial_message = self._build_initial_message(test_script)
            
            # Start generation with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Generation timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout_seconds))
            
            try:
                # Run conversation
                chat_result = self.tester.initiate_chat(
                    self.developer,
                    message=initial_message,
                    summary_method="reflection_with_llm"
                )
                
                # Check result
                success = self._verify_generation(chat_result)
                
                if success:
                    self.metrics.status = GenerationStatus.SUCCESS
                    self._finalize_model()
                    logger.info(f"✅ Successfully generated {self.model_name}")
                else:
                    self.metrics.status = GenerationStatus.FAILED
                    logger.error(f"Generation incomplete for {self.model_name}")
                
                return success
                
            finally:
                signal.alarm(0)  # Cancel timeout
                
        except TimeoutError:
            self.metrics.status = GenerationStatus.TIMEOUT
            logger.error(f"Generation timeout for {self.model_name}")
            return False
            
        except Exception as e:
            self.metrics.status = GenerationStatus.FAILED
            self.metrics.errors.append(str(e))
            logger.error(f"Generation failed: {e}")
            logger.debug(traceback.format_exc())
            return False
            
        finally:
            self.metrics.end_time = datetime.now()
            self._save_metrics()
            self._cleanup()
    
    def _build_initial_message(self, test_script: Path) -> str:
        """Build initial message for conversation"""
        return f"""
Please implement the {self.model_name} model following the MarketModel interface.

1. Create: models/{self.model_name}/{self.model_name}.py
2. Create: models/{self.model_name}/__init__.py with:
   from .{self.model_name} import {self.model_name}

3. Implement all required methods from the interface
4. Follow the paper architecture if provided, otherwise use LSTM

After implementation, run: python {test_script}

The tests will validate:
- Interface compliance
- Tensor shapes
- Training/prediction workflow
- State persistence

Keep iterating until ALL TESTS PASSED appears.
"""
    
    def _verify_generation(self, chat_result: Any) -> bool:
        """Verify the generated model"""
        # Check if model files exist
        model_file = self.model_dir / f"{self.model_name}.py"
        init_file = self.model_dir / "__init__.py"
        
        if not model_file.exists():
            logger.error(f"Model file not created: {model_file}")
            return False
        
        if not init_file.exists():
            logger.warning("__init__.py not created, creating default")
            init_file.write_text(f"from .{self.model_name} import {self.model_name}")
        
        # Validate code
        code = model_file.read_text()
        is_valid, errors = self.validator.validate_code(code)
        
        if not is_valid:
            logger.error(f"Code validation failed: {errors}")
            if not self.config.enable_code_validation:
                logger.warning("Validation disabled, continuing anyway")
            else:
                return False
        
        # Check conversation result
        if hasattr(chat_result, 'summary'):
            summary = str(chat_result.summary).upper()
            return "ALL TESTS PASSED" in summary or "SUCCESS" in summary
        
        return False
    
    def _finalize_model(self):
        """Finalize the generated model"""
        logger.info("Finalizing model")
        
        # Copy from workspace to models directory
        workspace_model = self.workspace_dir / "models" / self.model_name
        if workspace_model.exists():
            shutil.copytree(workspace_model, self.model_dir, dirs_exist_ok=True)
        
        # Create metadata
        metadata = {
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "source": {
                "paper": self.pdf_path,
                "github": self.github_url
            },
            "metrics": self.metrics.to_dict(),
            "config": {
                "llm_model": self.config.llm_model,
                "max_iterations": self.config.max_iterations
            }
        }
        
        metadata_file = self.model_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        # Run final validation
        self.validate_model()
    
    def validate_model(self) -> bool:
        """Run comprehensive validation"""
        logger.info(f"Validating {self.model_name}")
        
        self.metrics.status = GenerationStatus.VALIDATING
        self._update_health_status("validating")
        
        try:
            # Run full test suite
            result = subprocess.run(
                [sys.executable, "-m", "pytest",
                 "tests/test_model_interface.py",
                 "-v",
                 "--model_class", f"models.{self.model_name}.{self.model_name}.{self.model_name}",
                 "--timeout", "120"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            # Parse results
            output = result.stdout + result.stderr
            self.metrics.validation_results = {
                "all_tests": result.returncode == 0,
                "output_preview": output[-1000:]  # Last 1000 chars
            }
            
            if result.returncode == 0:
                logger.info("✅ All validation tests passed")
                return True
            else:
                logger.warning(f"Validation failed with code {result.returncode}")
                logger.debug(output)
                return False
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            self.metrics.validation_results = {"error": str(e)}
            return False
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save generation checkpoint for resumability"""
        checkpoint_dir = self.workspace_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict(),
            "model_exists": (self.model_dir / f"{self.model_name}.py").exists()
        }
        
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}_{int(time.time())}.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))
    
    def _save_metrics(self):
        """Save generation metrics"""
        metrics_file = self.workspace_dir / "metrics.json"
        metrics_file.write_text(json.dumps(self.metrics.to_dict(), indent=2))
    
    def _cleanup(self):
        """Cleanup temporary files"""
        if not self.config.save_conversations:
            try:
                shutil.rmtree(self.workspace_dir / "conversations", ignore_errors=True)
            except:
                pass
        
        self._update_health_status("completed")


class ParallelModelGenerator:
    """Generate multiple models in parallel"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_jobs
        )
    
    def generate_batch(
        self,
        models: List[Dict[str, str]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, bool]:
        """Generate multiple models with progress tracking"""
        
        logger.info(f"Starting batch generation of {len(models)} models")
        results = {}
        futures = {}
        
        # Submit jobs
        for model_config in models:
            future = self.executor.submit(
                self._generate_single,
                model_config
            )
            futures[future] = model_config['name']
        
        # Monitor progress
        completed = 0
        for future in futures:
            try:
                model_name = futures[future]
                success = future.result(timeout=self.config.timeout_seconds)
                results[model_name] = success
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(models), model_name, success)
                    
            except TimeoutError:
                model_name = futures[future]
                logger.error(f"Timeout generating {model_name}")
                results[model_name] = False
                
            except Exception as e:
                model_name = futures[future]
                logger.error(f"Error generating {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def _generate_single(self, model_config: Dict) -> bool:
        """Generate a single model"""
        try:
            generator = ModelGeneratorWorkflow(
                pdf_path=model_config.get('pdf_path'),
                github_url=model_config.get('github_url'),
                model_name=model_config['name'],
                config=self.config
            )
            return generator.generate_model()
        except Exception as e:
            logger.error(f"Failed to generate {model_config['name']}: {e}")
            return False


# Health check endpoint
def check_health(workspace_dir: Path = Path("temp")) -> Dict:
    """Check health of all active generations"""
    health_data = {}
    
    for model_dir in workspace_dir.iterdir():
        if model_dir.is_dir():
            health_file = model_dir / "health.json"
            if health_file.exists():
                try:
                    data = json.loads(health_file.read_text())
                    # Check if still active (updated within last hour)
                    last_update = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - last_update < timedelta(hours=1):
                        health_data[model_dir.name] = data
                except:
                    pass
    
    return health_data


# CLI entry point
def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate models from papers")
    parser.add_argument("--paper", type=str, help="Path to PDF paper")
    parser.add_argument("--github", type=str, help="GitHub repository URL")
    parser.add_argument("--name", type=str, required=True, help="Model name")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--health", action="store_true", help="Check health")
    
    args = parser.parse_args()
    
    if args.health:
        health = check_health()
        print(json.dumps(health, indent=2))
        return
    
    # Load config
    config = GenerationConfig()
    if args.config:
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Generate model
    generator = ModelGeneratorWorkflow(
        pdf_path=args.paper,
        github_url=args.github,
        model_name=args.name,
        config=config
    )
    
    success = generator.generate_model()
    
    if success and args.validate:
        generator.validate_model()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()