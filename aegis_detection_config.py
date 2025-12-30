#!/usr/bin/env python3
"""
Aegis Insight - Detection Configuration System

Provides user-configurable detection parameters with persistent storage.
Allows users to adjust detection sensitivity without editing code.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Configuration file location
CONFIG_FILE = Path.home() / ".aegis" / "detection_config.json"


@dataclass
class DetectionConfig:
    """
    User-configurable detection parameters
    
    All detectors use these settings instead of hardcoded values.
    Changes persist across restarts.
    """
    
    # =========================================================================
    # Claim Analysis Limits
    # =========================================================================
    min_claims_analyzed: int = 25
    """Minimum number of claims to analyze (soft limit)"""
    
    max_claims_analyzed: int = 200
    """Maximum number of claims to analyze"""
    
    # =========================================================================
    # Suppression Detector Thresholds
    # =========================================================================
    citation_asymmetry_threshold: float = 0.6
    """Citation imbalance threshold (0.0-1.0). Higher = more asymmetry required"""
    
    meta_dismissal_weight: float = 0.8
    """Weight given to META dismissals in suppression score (0.0-1.0)"""
    
    require_institutional_response: bool = False
    """If True, require META claims for suppression detection"""
    
    # =========================================================================
    # Coordination Detector Thresholds
    # =========================================================================
    language_similarity_threshold: float = 0.75
    """Minimum semantic similarity for coordinated messaging (0.0-1.0)"""
    
    temporal_clustering_threshold: float = 0.7
    """Threshold for temporal clustering detection (0.0-1.0)"""
    
    min_sources_for_coordination: int = 3
    """Minimum number of sources required to detect coordination"""
    
    # =========================================================================
    # Anomaly Detector Thresholds
    # =========================================================================
    geographic_diversity_required: int = 2
    """Minimum number of geographic contexts for anomaly detection"""
    
    cultural_context_weight: float = 0.6
    """Weight given to cultural context in anomaly scoring (0.0-1.0)"""
    
    min_anomaly_strength: float = 0.5
    """Minimum anomaly strength to report (0.0-1.0)"""
    
    def save(self) -> bool:
        """
        Save configuration to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Write config
            with open(CONFIG_FILE, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            
            logger.info(f"Detection config saved to {CONFIG_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save detection config: {e}")
            return False
    
    @classmethod
    def load(cls) -> 'DetectionConfig':
        """
        Load configuration from disk or return defaults
        
        Returns:
            DetectionConfig instance
        """
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                
                # Validate and create config
                config = cls(**data)
                logger.info(f"Detection config loaded from {CONFIG_FILE}")
                return config
                
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
                return cls()
        else:
            logger.info("No config file found, using defaults")
            return cls()
    
    @classmethod
    def reset_defaults(cls) -> 'DetectionConfig':
        """
        Reset configuration to default values and save
        
        Returns:
            New DetectionConfig with defaults
        """
        config = cls()
        config.save()
        logger.info("Detection config reset to defaults")
        return config
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration values
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        # Check ranges
        if not (0 < self.min_claims_analyzed <= 1000):
            errors.append("min_claims_analyzed must be between 1 and 1000")
        
        if not (self.min_claims_analyzed <= self.max_claims_analyzed <= 1000):
            errors.append("max_claims_analyzed must be >= min and <= 1000")
        
        # Check thresholds (0.0-1.0)
        for field in ['citation_asymmetry_threshold', 'meta_dismissal_weight',
                      'language_similarity_threshold', 'temporal_clustering_threshold',
                      'cultural_context_weight', 'min_anomaly_strength']:
            value = getattr(self, field)
            if not (0.0 <= value <= 1.0):
                errors.append(f"{field} must be between 0.0 and 1.0")
        
        # Check minimums
        if self.min_sources_for_coordination < 2:
            errors.append("min_sources_for_coordination must be >= 2")
        
        if self.geographic_diversity_required < 1:
            errors.append("geographic_diversity_required must be >= 1")
        
        return (len(errors) == 0, errors)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DetectionConfig(claims: {self.min_claims_analyzed}-{self.max_claims_analyzed})"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_config() -> DetectionConfig:
    """
    Get current detection configuration
    
    This is the primary function used by detectors.
    
    Returns:
        DetectionConfig instance
    """
    return DetectionConfig.load()


def update_config(**kwargs) -> tuple[bool, Optional[str]]:
    """
    Update specific configuration values
    
    Args:
        **kwargs: Field names and new values
    
    Returns:
        (success, error_message)
    
    Example:
        update_config(max_claims_analyzed=300, citation_asymmetry_threshold=0.7)
    """
    try:
        # Load current config
        config = get_config()
        
        # Update provided fields
        for key, value in kwargs.items():
            if not hasattr(config, key):
                return False, f"Unknown config field: {key}"
            setattr(config, key, value)
        
        # Validate
        is_valid, errors = config.validate()
        if not is_valid:
            return False, "; ".join(errors)
        
        # Save
        if config.save():
            return True, None
        else:
            return False, "Failed to save config"
        
    except Exception as e:
        return False, str(e)


def reset_config() -> DetectionConfig:
    """
    Reset to default configuration
    
    Returns:
        New config with defaults
    """
    return DetectionConfig.reset_defaults()


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "show":
            config = get_config()
            print("\nCurrent Detection Configuration:")
            print("=" * 60)
            for key, value in asdict(config).items():
                print(f"{key:40} {value}")
            print("=" * 60)
        
        elif command == "reset":
            config = reset_config()
            print("✓ Configuration reset to defaults")
        
        elif command == "set" and len(sys.argv) >= 4:
            field = sys.argv[2]
            value = sys.argv[3]
            
            # Try to parse value
            try:
                if '.' in value:
                    value = float(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                else:
                    value = int(value)
            except ValueError:
                pass
            
            success, error = update_config(**{field: value})
            if success:
                print(f"✓ Updated {field} = {value}")
            else:
                print(f"✗ Error: {error}")
        
        else:
            print("Usage:")
            print("  python aegis_detection_config.py show")
            print("  python aegis_detection_config.py reset")
            print("  python aegis_detection_config.py set <field> <value>")
    else:
        # Demo
        print("Detection Configuration System")
        print("=" * 60)
        
        config = get_config()
        print(f"\nCurrent: {config}")
        print(f"Config file: {CONFIG_FILE}")
        
        print("\nValidation:")
        is_valid, errors = config.validate()
        if is_valid:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
