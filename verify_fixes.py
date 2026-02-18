"""Verification script to check if both Docker and local environments match.

Run this script in both environments to compare:
1. Python version
2. Package versions (LightGBM, PyMuPDF, etc.)
3. Model file checksums
4. PyMuPDF availability
5. Sample prediction consistency
"""
import sys
import hashlib
from pathlib import Path
import json

def get_checksum(filepath, bytes_to_read=10000):
    """Get MD5 checksum of first N bytes of file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read(bytes_to_read)).hexdigest()[:8]
    except Exception as e:
        return f"ERROR: {e}"

def check_environment():
    """Check environment details."""
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Python version
    print(f"\n1. Python Version: {sys.version}")
    
    # Package versions
    print("\n2. Package Versions:")
    packages = {
        'numpy': ('numpy', '__version__'),
        'lightgbm': ('lightgbm', '__version__'),
        'torch': ('torch', '__version__'),
        'transformers': ('transformers', '__version__'),
        'PIL': ('PIL', '__version__'),
        'pymupdf': ('fitz', '__version__'),
    }
    
    for name, (module, attr) in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, attr, 'unknown')
            print(f"   {name}: {version} ✓")
        except ImportError:
            print(f"   {name}: NOT INSTALLED ✗")
    
    # Model file checksums
    print("\n3. Model File Checksums:")
    artifacts_dir = Path("artifacts")
    model_files = [
        "sensor_baseline.pkl",
        "joint_sensor_image.pkl",
        "transformer_fusion_trained.pt",
    ]
    
    for model_file in model_files:
        path = artifacts_dir / model_file
        if path.exists():
            checksum = get_checksum(path)
            size = path.stat().st_size
            print(f"   {model_file}: {checksum} ({size:,} bytes) ✓")
        else:
            print(f"   {model_file}: NOT FOUND ✗")
    
    # PyMuPDF functionality
    print("\n4. PyMuPDF PDF Processing:")
    try:
        import fitz as pymupdf
        print(f"   PyMuPDF installed: YES ✓")
        print(f"   Version: {pymupdf.__version__}")
    except ImportError:
        print(f"   PyMuPDF installed: NO ✗")
        print(f"   → PDFs will NOT work without this!")
    
    # Random seed test
    print("\n5. Random Seed Test (should be identical in both environments):")
    import numpy as np
    np.random.seed(42)
    random_values = np.random.rand(5)
    print(f"   First 5 random values: {random_values}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

def test_prediction():
    """Test a simple prediction."""
    print("\n6. Sample Prediction Test:")
    try:
        from src.services.orchestrator import InferenceOrchestrator
        from src.api.schemas.request import PredictionRequest
        
        # Create sample request
        request = PredictionRequest(
            asset_id="test_pump",
            timestamp="2026-02-18T10:00:00Z",
            sensor_window=[{
                f"sensor_{i:02d}": 45.0 + i * 0.5 
                for i in range(52)
            }]
        )
        
        orchestrator = InferenceOrchestrator()
        import asyncio
        result = asyncio.run(orchestrator.predict(request))
        
        print(f"\n   Asset ID: {result.asset_id}")
        print(f"   Failure Probability: {result.failure_probability:.4f}")
        print(f"   Confidence: {result.fault_confidence:.4f}")
        print(f"   Top Signals: {', '.join(result.top_signals[:3])}")
        print(f"   Inference Time: {result.inference_ms}ms")
        
        # Save result for comparison
        result_data = {
            "failure_probability": result.failure_probability,
            "fault_confidence": result.fault_confidence,
            "top_signals": result.top_signals,
        }
        
        with open("verification_result.json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n   ✓ Result saved to verification_result.json")
        print(f"   → Compare this file between Docker and local!")
        
    except Exception as e:
        print(f"\n   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_environment()
    test_prediction()
