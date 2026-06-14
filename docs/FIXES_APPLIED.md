# Issues Fixed - Docker vs Local Predictions & PDF Processing

## Problems Identified

### 1. **"Insufficient data" for PDFs in Docker** ❌
- **Root Cause**: `PyMuPDF` was only in `requirements-dev.txt`, NOT in `requirements.txt`
- **Impact**: Docker containers couldn't extract images from PDFs
- **Result**: When PDFs were uploaded without sensor data, fusion returned "insufficient_data"

### 2. **Different predictions Docker vs Local** ❌
- **Root Cause**: No random seeds set → non-deterministic model initialization
- **Impact**: Same input gave different outputs across environments
- **Result**: Inconsistent predictions between Docker and local environments

## Solutions Applied ✅

### Fix 1: Add PyMuPDF to Production Requirements
**File**: `requirements.txt`
```diff
 # Image processing
 Pillow>=10.0.0
+PyMuPDF>=1.23.0
 
 # HTTP client
```

### Fix 2: Add Random Seeding for Reproducibility
**File**: `src/services/orchestrator.py`
```python
# Set random seeds for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# Also set for torch if available
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass
```

### Fix 3: Enhanced Logging & Error Messages
**Files**: 
- `src/services/orchestrator.py` - Better PDF extraction logging
- `src/models/risk_model.py` - Model checksums for debugging
- `src/models/fusion.py` - Clearer "no input data" messages

**Key improvements**:
- Logs model file checksums to verify artifacts match
- Warns when PyMuPDF is missing
- Reports PDF page count and extracted image count
- Shows clear error when no data is provided

### Fix 4: Better Error Handling
- Silent failures now log warnings
- PDF extraction failures show actionable messages
- Model loading shows file size and checksum for verification

## How to Verify the Fixes

### Step 1: Rebuild Docker Container
```bash
# Stop existing containers
docker-compose down

# Rebuild with updated requirements.txt (includes PyMuPDF now)
docker-compose build --no-cache

# Start services
docker-compose up
```

### Step 2: Run Verification Script (Local)
```bash
# In your local environment
python verify_fixes.py
```

This will output:
- Python and package versions
- Model file checksums
- PyMuPDF availability
- Sample prediction result saved to `verification_result.json`

### Step 3: Run Verification Script (Docker)
```bash
# Inside Docker container
docker-compose exec app python verify_fixes.py
```

### Step 4: Compare Results
```bash
# Compare the verification_result.json from both environments
# They should now be IDENTICAL

# Local result:
cat verification_result.json

# Docker result (run inside container):
docker-compose exec app cat verification_result.json
```

## Expected Outcomes

### Before Fixes ❌
| Test | Local | Docker | Match? |
|------|-------|--------|--------|
| PDF extraction | ✓ Works | ✗ Silent fail | ❌ Different |
| Prediction with PDF only | ✓ Works | ✗ "insufficient data" | ❌ Different |
| Same sensor input | 0.1234 | 0.5678 | ❌ Different |

### After Fixes ✅
| Test | Local | Docker | Match? |
|------|-------|--------|--------|
| PDF extraction | ✓ Works | ✓ Works | ✅ Same |
| Prediction with PDF only | ✓ Works | ✓ Works | ✅ Same |
| Same sensor input | 0.1234 | 0.1234 | ✅ Same |

## Testing PDFs

### Test 1: PDF with Sensor Data
```bash
curl -X POST http://localhost:8000/predict/multimodal \
  -F "asset_id=pump_test" \
  -F "pdfs=@test_document.pdf" \
  -F 'sensor_json=[{"sensor_00": 45.0, "sensor_01": 50.0}]'
```

**Expected**: Should work in both Docker and local, same predictions

### Test 2: PDF Only (No Sensor Data)
```bash
curl -X POST http://localhost:8000/predict/multimodal \
  -F "asset_id=pump_test" \
  -F "pdfs=@test_document.pdf"
```

**Before fix**: Docker returns "insufficient_data", local works
**After fix**: Both work if PDF contains images

### Test 3: Check Logs
```bash
# Docker logs should now show:
docker-compose logs | grep -i "pymupdf\|pdf\|checksum"

# You should see:
# - "Processing PDF with X pages"
# - "Extracted N images from PDF"
# - Model checksums matching local environment
```

## Troubleshooting

### Issue: Still seeing "insufficient_data"
**Check**:
1. Rebuild Docker: `docker-compose build --no-cache`
2. Verify PyMuPDF installed: `docker-compose exec app pip list | grep PyMuPDF`
3. Check logs: `docker-compose logs app | grep -i pdf`

### Issue: Still different predictions
**Check**:
1. Model checksums match: Run `verify_fixes.py` in both environments
2. Compare `verification_result.json` files
3. Check if using same artifacts: `ls -lh artifacts/`

### Issue: PDF has no images
Some PDFs are pure text with no embedded images. In this case:
- System correctly reports: "No images found in PDF document"
- Must provide sensor_window for prediction
- This is expected behavior

## Summary

✅ **PyMuPDF now in requirements.txt** - Docker can process PDFs
✅ **Random seeds set** - Predictions are now deterministic
✅ **Better logging** - Easy to debug environment differences
✅ **Model checksums** - Verify artifacts match across environments
✅ **Clearer errors** - Know exactly what data is missing

Your Docker and local environments should now produce **identical predictions** for the same inputs!
