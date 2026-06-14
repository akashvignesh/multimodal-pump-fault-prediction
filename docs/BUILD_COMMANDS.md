# Docker Build with Version Backup - Simple Commands

## Execute these commands one by one:

### 1. Backup Current Image (Before Fixes)
```powershell
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.0-before-fixes
```

Verify backup was created:
```powershell
docker images | Select-String "pump-fault"
```

### 2. Stop Current Containers
```powershell
docker-compose down
```

### 3. Build New Image with Fixes (includes PyMuPDF + seeding)
```powershell
docker-compose build --no-cache
```
*This will take 5-10 minutes. Be patient!*

### 4. Tag New Image
```powershell
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.1-with-pdf-fixes
```

### 5. Start New Version
```powershell
docker-compose up -d
```

### 6. Verify New Version is Running
```powershell
# Check logs for success indicators
docker-compose logs | Select-String -Pattern "PyMuPDF|checksum|Orchestrator initialized"

# Test health endpoint
curl http://localhost:8000/health

# Run verification inside container
docker-compose exec app python verify_fixes.py
```

---

## If New Version Has Problems - ROLLBACK:

### Stop New Version
```powershell
docker-compose down
```

### Revert to Old Version
```powershell
docker tag pump-fault-risk-service-app:v1.0.0-before-fixes pump-fault-risk-service-app:latest
```

### Start Old Version
```powershell
docker-compose up -d
```

---

## If New Version Works - SWITCH BACK to New:

### Stop Old Version
```powershell
docker-compose down
```

### Switch to New Version
```powershell
docker tag pump-fault-risk-service-app:v1.0.1-with-pdf-fixes pump-fault-risk-service-app:latest
```

### Start New Version
```powershell
docker-compose up -d
```

---

## List All Available Versions
```powershell
docker images | Select-String "pump-fault"
```

You should see:
- `latest` - Currently active version
- `v1.0.0-before-fixes` - OLD backup (without PyMuPDF)
- `v1.0.1-with-pdf-fixes` - NEW version (with fixes)

---

## Test PDF Functionality

Once new version is running, test PDF upload:

```powershell
# Create a test request (using Streamlit UI or curl)
# Go to http://localhost:8501 and upload a PDF with images
# OR use curl:

curl -X POST http://localhost:8000/predict/multimodal `
  -F "asset_id=test_pump" `
  -F "pdfs=@path\to\your\document.pdf"
```

Expected in logs:
- ✅ "Processing PDF with X pages"
- ✅ "Extracted N images from PDF"
- ✅ Prediction returns with results (not "insufficient_data")

---

## Cleanup (After Confirming New Version Works)

Remove old version to save disk space:
```powershell
docker rmi pump-fault-risk-service-app:v1.0.0-before-fixes
```

**⚠️ Only do this after thoroughly testing the new version!**
