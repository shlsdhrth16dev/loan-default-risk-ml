"""
Lightweight Model Versioning System.

Provides Git-like model versioning without MLflow:
- Semantic versioning (major.minor.patch)
- Model registry with metadata
- Version comparison
- Rollback capabilities
- Production/staging tags
"""
import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelRegistry:
    """
    Lightweight model registry for version management.
    
    Directory structure:
    models/
    ├── registry/
    │   ├── v1.0.0/
    │   │   ├── model.pkl
    │   │   ├── metadata.json
    │   │   └── metrics.json
    │   ├── v1.1.0/
    │   └── v2.0.0/
    ├── production -> registry/v1.0.0  (symlink)
    ├── staging -> registry/v1.1.0
    └── registry.json  (version index)
    """
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.registry_dir = self.base_dir / "registry"
        self.registry_file = self.base_dir / "registry.json"
        
        # Create directories
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create registry index
        self.registry_index = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry index."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            "versions": {},
            "production": None,
            "staging": None,
            "latest": None
        }
    
    def _save_registry(self):
        """Save registry index."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry_index, f, indent=2)
    
    def _parse_version(self, version: str) -> tuple:
        """Parse semantic version string."""
        parts = version.lstrip('v').split('.')
        return tuple(int(p) for p in parts)
    
    def _bump_version(
        self, 
        current: Optional[str],
        bump_type: str = "patch"
    ) -> str:
        """
        Bump version number.
        
        Args:
            current: Current version (e.g., "v1.0.0")
            bump_type: "major", "minor", or "patch"
        
        Returns:
            New version string
        """
        if current is None:
            return "v1.0.0"
        
        major, minor, patch = self._parse_version(current)
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"v{major}.{minor}.{patch}"
    
    def register_model(
        self,
        model_path: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, float],
        version: Optional[str] = None,
        bump_type: str = "patch",
        description: str = ""
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to model file
            metadata: Model metadata (hyperparams, features, etc.)
            metrics: Model performance metrics
            version: Explicit version (or auto-bump)
            bump_type: How to bump version if auto
            description: Version description
        
        Returns:
            Version string
        """
        # Determine version
        if version is None:
            latest = self.registry_index.get("latest")
            version = self._bump_version(latest, bump_type)
        
        # Ensure version format
        if not version.startswith('v'):
            version = f'v{version}'
        
        print(f"Registering model version {version}...")
        
        # Create version directory
        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        model_dest = version_dir / "model.pkl"
        shutil.copy2(model_path, model_dest)
        
        # Save metadata
        metadata_full = {
            **metadata,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "description": description,
            "model_path": str(model_dest)
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata_full, f, indent=2)
        
        # Save metrics
        with open(version_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Update registry index
        self.registry_index["versions"][version] = {
            "registered_at": metadata_full["registered_at"],
            "description": description,
            "metrics": metrics,
            "path": str(version_dir)
        }
        self.registry_index["latest"] = version
        
        self._save_registry()
        
        print(f"✓ Model {version} registered successfully")
        return version
    
    def promote_to_production(self, version: str):
        """Promote a version to production."""
        if not version.startswith('v'):
            version = f'v{version}'
        
        if version not in self.registry_index["versions"]:
            raise ValueError(f"Version {version} not found in registry")
        
        self.registry_index["production"] = version
        self._save_registry()
        
        # Create/update symlink (Windows: update registry instead)
        print(f"✓ Version {version} promoted to PRODUCTION")
    
    def promote_to_staging(self, version: str):
        """Promote a version to staging."""
        if not version.startswith('v'):
            version = f'v{version}'
        
        if version not in self.registry_index["versions"]:
            raise ValueError(f"Version {version} not found in registry")
        
        self.registry_index["staging"] = version
        self._save_registry()
        
        print(f"✓ Version {version} promoted to STAGING")
    
    def get_model(self, version: Optional[str] = None) -> tuple:
        """
        Load model by version.
        
        Args:
            version: Version to load (default: production)
        
        Returns:
            (model, metadata, metrics)
        """
        if version is None:
            version = self.registry_index.get("production")
            if version is None:
                raise ValueError("No production model set")
        
        if not version.startswith('v'):
            version = f'v{version}'
        
        if version not in self.registry_index["versions"]:
            raise ValueError(f"Version {version} not found")
        
        version_dir = self.registry_dir / version
        
        # Load model
        with open(version_dir / "model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(version_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load metrics
        with open(version_dir / "metrics.json", 'r') as f:
            metrics = json.load(f)
        
        return model, metadata, metrics
    
    def list_versions(self) -> List[Dict]:
        """List all registered versions."""
        versions = []
        for version, info in self.registry_index["versions"].items():
            version_info = {
                "version": version,
                **info,
                "is_production": version == self.registry_index.get("production"),
                "is_staging": version == self.registry_index.get("staging"),
                "is_latest": version == self.registry_index.get("latest")
            }
            versions.append(version_info)
        
        # Sort by version number
        versions.sort(
            key=lambda x: self._parse_version(x["version"]),
            reverse=True
        )
        
        return versions
    
    def compare_versions(
        self, 
        version1: str, 
        version2: str
    ) -> Dict:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Comparison dictionary
        """
        if not version1.startswith('v'):
            version1 = f'v{version1}'
        if not version2.startswith('v'):
            version2 = f'v{version2}'
        
        # Load metrics
        metrics1_path = self.registry_dir / version1 / "metrics.json"
        metrics2_path = self.registry_dir / version2 / "metrics.json"
        
        with open(metrics1_path, 'r') as f:
            metrics1 = json.load(f)
        
        with open(metrics2_path, 'r') as f:
            metrics2 = json.load(f)
        
        # Compare
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {}
        }
        
        for metric, value1 in metrics1.items():
            value2 = metrics2.get(metric)
            if value2 is not None:
                diff = value2 - value1
                pct_change = (diff / value1 * 100) if value1 != 0 else 0
                
                comparison["metrics_comparison"][metric] = {
                    "v1": value1,
                    "v2": value2,
                    "diff": diff,
                    "pct_change": pct_change,
                    "better": "v2" if diff > 0 else "v1" if diff < 0 else "equal"
                }
        
        return comparison
    
    def rollback_production(self):
        """Rollback production to previous version."""
        current_prod = self.registry_index.get("production")
        if current_prod is None:
            raise ValueError("No production model to rollback")
        
        # Find previous version
        versions = sorted(
            self.registry_index["versions"].keys(),
            key=lambda x: self._parse_version(x),
            reverse=True
        )
        
        current_idx = versions.index(current_prod)
        if current_idx >= len(versions) - 1:
            raise ValueError("No previous version to rollback to")
        
        previous_version = versions[current_idx + 1]
        
        print(f"Rolling back production from {current_prod} to {previous_version}")
        self.promote_to_production(previous_version)
        
        return previous_version
    
    def get_info(self) -> Dict:
        """Get registry information."""
        return {
            "total_versions": len(self.registry_index["versions"]),
            "production": self.registry_index.get("production"),
            "staging": self.registry_index.get("staging"),
            "latest": self.registry_index.get("latest"),
            "versions": self.list_versions()
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("MODEL VERSIONING SYSTEM - DEMO")
    print("="*70)
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Example: Register current production model
    metadata = {
        "model_type": "XGBoost",
        "n_features": 29,
        "training_samples": 255347,
        "hyperparameters": {
            "n_estimators": 680,
            "max_depth": 3,
            "learning_rate": 0.0255
        }
    }
    
    metrics = {
        "roc_auc": 0.7605,
        "precision": 0.2942,
        "recall": 0.5249,
        "f1": 0.3770
    }
    
    # Register
    version = registry.register_model(
        model_path="models/production/loan_default_model_final.pkl",
        metadata=metadata,
        metrics=metrics,
        version="v1.0.0",
        description="Initial production model with tuned hyperparameters"
    )
    
    # Promote to production
    registry.promote_to_production(version)
    
    # Show info
    info = registry.get_info()
    print(f"\nRegistry Info:")
    print(f"  Total versions: {info['total_versions']}")
    print(f"  Production: {info['production']}")
    print(f"  Latest: {info['latest']}")
    
    print("\n✓ Model versioning system ready!")
