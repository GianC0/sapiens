"""
DeepCode integration for automatic model generation from papers/repos.
"""

import os
import re
import ast
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ModelGenerator:
    """
    Generates SapiensModel implementations from research papers or GitHub repos using DeepCode.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with DeepCode params
        """
        self.config = config
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.anthropic_key)
        self.models_dir = Path("models")
        
    def generate_model(
        self,
        source_type: str,
        source_path: str,
        model_name: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate a SapiensModel from source.
        
        Args:
            source_type: "github", "pdf", or "arxiv"
            source_path: URL or file path to source
            model_name: Name for generated model
            output_dir: Optional custom output directory
            
        Returns:
            Path to generated model directory
        """
        logger.info(f"Generating {model_name} from {source_type}: {source_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.models_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract model architecture from source
        architecture_info = self._extract_architecture(source_type, source_path)
        
        # Step 2: Generate SapiensModel implementation
        model_code = self._generate_sapiens_model(model_name, architecture_info)
        
        # Step 3: Write model file
        model_file = output_dir / f"{model_name}.py"
        model_file.write_text(model_code)
        logger.info(f"Generated model code: {model_file}")
        
        # Step 4: Extract parameters and generate config
        params, hparams = self._extract_parameters(model_code)
        config_content = self._generate_config_yaml(model_name, params, hparams)
        
        config_file = output_dir / "model_config.yaml"
        config_file.write_text(config_content)
        logger.info(f"Generated config: {config_file}")
        
        # Step 5: Create __init__.py
        init_file = output_dir / "__init__.py"
        init_file.write_text(f"from .{model_name} import {model_name}\n\n__all__ = ['{model_name}']\n")
        
        logger.info(f"✅ Model generation complete: {output_dir}")
        return output_dir
    
    def _extract_architecture(self, source_type: str, source_path: str) -> Dict[str, Any]:
        """Extract model architecture information from source."""
        
        system_prompt = """You are an expert at analyzing machine learning research papers and code.
Extract the following information about the model architecture:
1. Input features and dimensions
2. Model components (layers, attention mechanisms, etc.)
3. Forward pass logic
4. Loss functions
5. Training hyperparameters
6. Any special preprocessing or data requirements

Provide a structured JSON response."""
        
        if source_type == "github":
            content = self._fetch_github_content(source_path)
        elif source_type == "pdf":
            content = self._read_pdf(source_path)
        elif source_type == "arxiv":
            content = self._fetch_arxiv(source_path)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")
        
        user_prompt = f"Analyze this {source_type} source and extract model architecture:\n\n{content[:20000]}"
        
        response = self.client.messages.create(
            model=self.config.get("claude_model", "claude-sonnet-4-20250514"),
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        architecture_text = response.content[0].text
        # Parse JSON from response
        import json
        try:
            architecture_info = json.loads(architecture_text)
        except json.JSONDecodeError:
            # Fallback: use text as-is
            architecture_info = {"raw_description": architecture_text}
        
        return architecture_info
    
    def _generate_sapiens_model(self, model_name: str, architecture_info: Dict) -> str:
        """Generate SapiensModel-compliant code."""
        
        # Read SapiensModel interface
        sapiens_model_path = self.models_dir / "SapiensModel.py"
        sapiens_interface = sapiens_model_path.read_text()
        
        system_prompt = f"""You are an expert PyTorch developer specializing in financial ML models.

Generate a complete SapiensModel implementation that:
1. Inherits from SapiensModel base class
2. Implements _build_model(), _forward_train(), _forward_predict() methods
3. Follows this interface:

{sapiens_interface}

Key requirements:
- Use descriptive parameter names (no magic numbers)
- All hyperparameters in _default_hparams() method
- Proper tensor shapes and device handling
- Return predictions as Dict[str, float] in _forward_predict()
- Return scalar loss in _forward_train()

Output ONLY valid Python code, no markdown or explanations."""
        
        user_prompt = f"""Generate a SapiensModel implementation named '{model_name}' based on:

Architecture: {architecture_info}

Include:
1. Class definition inheriting from SapiensModel
2. _default_hparams() with all tunable hyperparameters
3. _build_model() building the architecture
4. _forward_train() computing loss
5. _forward_predict() returning predictions
6. All necessary imports"""
        
        response = self.client.messages.create(
            model=self.config.get("claude_model", "claude-sonnet-4-20250514"),
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        code = response.content[0].text
        
        # Strip markdown if present
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        
        return code
    
    def _extract_parameters(self, model_code: str) -> tuple[Dict, Dict]:
        """
        Extract parameters and hyperparameters from generated code.
        
        Returns:
            (params, hparams) dictionaries
        """
        # Parse AST
        tree = ast.parse(model_code)
        
        params = {}
        hparams = {}
        
        # Find __init__ and _default_hparams methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    # Extract parameters from __init__ signature
                    for arg in node.args.args:
                        if arg.arg not in ['self', 'config']:
                            params[arg.arg] = None  # Will be filled from config
                
                elif node.name == "_default_hparams":
                    # Extract hyperparameters from return dict
                    for stmt in node.body:
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                            for key, value in zip(stmt.value.keys, stmt.value.values):
                                if isinstance(key, ast.Constant):
                                    hparam_name = key.value
                                    # Get default value
                                    if isinstance(value, ast.Constant):
                                        hparams[hparam_name] = value.value
                                    else:
                                        hparams[hparam_name] = None
        
        # Add standard params
        standard_params = {
            'model_name': 'AUTO',
            'feature_dim': 5,
            'window_len': 20,
            'pred_len': 1,
            'target_idx': 3,
            'n_epochs': 50,
            'batch_size': 32,
            'patience': 5,
            'warm_start': True,
            'save_backups': False,
        }
        
        params.update(standard_params)
        
        return params, hparams
    
    def _generate_config_yaml(self, model_name: str, params: Dict, hparams: Dict) -> str:
        """Generate model_config.yaml for the model."""
        
        # Build PARAMS section
        params_section = {
            'model_name': model_name,
            'logs_dir': 'logs/models/',
            'feature_dim': params.get('feature_dim', 5),
            'features_to_load': 'candles',
            'adjust': True,
            'pred_len': params.get('pred_len', 1),
            'target_idx': params.get('target_idx', 3),
            'n_epochs': params.get('n_epochs', 50),
            'batch_size': params.get('batch_size', 32),
            'warm_start': params.get('warm_start', True),
            'patience': params.get('patience', 5),
            'save_backups': params.get('save_backups', False),
        }
        
        # Build HPARAMS section with Optuna config
        hparams_section = {}
        for hparam_name, default_value in hparams.items():
            if isinstance(default_value, float):
                # Assume log-scale for learning rates
                if 'lr' in hparam_name or 'learning_rate' in hparam_name:
                    hparams_section[hparam_name] = {
                        'default': default_value,
                        'optuna': {
                            'optuna_type': 'log_low-high',
                            'low': default_value / 10,
                            'high': default_value * 10,
                        }
                    }
                else:
                    hparams_section[hparam_name] = {
                        'default': default_value,
                        'optuna': {
                            'optuna_type': 'low-high',
                            'low': default_value * 0.5,
                            'high': default_value * 2.0,
                        }
                    }
            elif isinstance(default_value, int):
                hparams_section[hparam_name] = {
                    'default': default_value,
                    'optuna': {
                        'optuna_type': 'int_low-high',
                        'low': max(1, default_value // 2),
                        'high': default_value * 2,
                    }
                }
            else:
                # Categorical or other
                hparams_section[hparam_name] = {
                    'default': default_value
                }
        
        config = {
            'MODEL': {
                'PARAMS': params_section,
                'HPARAMS': hparams_section
            }
        }
        
        # Add comments
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        
        header = f"""# Auto-generated config for {model_name}
# Generated by ModelGenerator using DeepCode
#
# PARAMS: Fixed model parameters
# HPARAMS: Hyperparameters for Optuna optimization
#
"""
        return header + yaml_str
    
    def _fetch_github_content(self, repo_url: str) -> str:
        """Fetch README and key files from GitHub repo."""
        # Simple implementation - fetch README
        import requests
        
        # Convert github.com URL to raw.githubusercontent.com
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]
        
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
        try:
            response = requests.get(readme_url)
            return response.text
        except:
            # Try master branch
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
            response = requests.get(readme_url)
            return response.text
    
    def _read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        import PyPDF2
        
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def _fetch_arxiv(self, arxiv_id: str) -> str:
        """Fetch paper from arXiv."""
        import requests
        
        # Extract ID from URL if needed
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1].replace(".pdf", "")
        
        # Fetch abstract and metadata
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(api_url)
        return response.text