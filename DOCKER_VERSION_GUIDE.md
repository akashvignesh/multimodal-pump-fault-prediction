# Docker Version Management Guide

## Quick Start

### Windows (PowerShell)
```powershell
# Run the automated script
.\docker-version-management.ps1
```

### Linux/Mac (Bash)
```bash
# Make executable and run
chmod +x docker-version-management.sh
./docker-version-management.sh
```

---

## Manual Step-by-Step Instructions

### Step 1: Backup Current Docker Image

Before building the new version, tag your current working image:

```bash
# Check current images
docker images | grep pump-fault-risk

# Tag current image as backup (before fixes)
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.0-before-fixes

# Verify backup was created
docker images | grep pump-fault-risk
```

### Step 2: Build New Version with Fixes

```bash
# Stop current containers
docker-compose down

# Build new image (includes PyMuPDF and random seeding fixes)
docker-compose build --no-cache

# Tag new image
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.1-with-pdf-fixes

# Start new version
docker-compose up -d
```

### Step 3: Test New Version

```bash
# Check logs
docker-compose logs -f

# Look for these SUCCESS indicators:
# ✓ "PyMuPDF installed"
# ✓ "Loaded sensor baseline model" with checksum
# ✓ "Processing PDF with X pages" (when testing PDFs)
# ✓ "Extracted N images from PDF"

# Test health endpoint
curl http://localhost:8000/health

# Run verification
docker-compose exec app python verify_fixes.py
```

### Step 4: Compare Results

```bash
# Save Docker result
docker-compose exec app python verify_fixes.py
docker-compose exec app cat verification_result.json > docker_result.json

# Compare with local result
python verify_fixes.py
# Compare docker_result.json with verification_result.json
# They should be IDENTICAL now
```

---

## Version Management Commands

### List All Available Versions
```bash
docker images | grep pump-fault-risk
```

Expected output:
```
pump-fault-risk-service-app   latest                 abc123def456   2 minutes ago   2.1GB
pump-fault-risk-service-app   v1.0.1-with-pdf-fixes  abc123def456   2 minutes ago   2.1GB
pump-fault-risk-service-app   v1.0.0-before-fixes    xyz789abc012   1 day ago       2.0GB
```

### Rollback to Old Version

If the new version has issues:

```bash
# Stop new version
docker-compose down

# Revert to old version
docker tag pump-fault-risk-service-app:v1.0.0-before-fixes pump-fault-risk-service-app:latest

# Start old version
docker-compose up -d

# Verify you're running old version
docker ps
docker-compose logs | grep "Model checksum"  # Should match old checksum
```

### Switch to New Version Again

After testing and confirming old version works:

```bash
# Stop old version
docker-compose down

# Switch to new version
docker tag pump-fault-risk-service-app:v1.0.1-with-pdf-fixes pump-fault-risk-service-app:latest

# Start new version
docker-compose up -d
```

---

## Version Comparison

| Feature | v1.0.0 (Before) | v1.0.1 (After) |
|---------|----------------|----------------|
| PyMuPDF in requirements.txt | ❌ No | ✅ Yes |
| PDF image extraction | ❌ Fails silently | ✅ Works + logs |
| Random seeding | ❌ No (non-deterministic) | ✅ Yes (seed=42) |
| Model checksums logged | ❌ No | ✅ Yes |
| Prediction consistency | ❌ Varies | ✅ Deterministic |
| PDF-only uploads | ❌ "insufficient_data" | ✅ Works |

---

## Troubleshooting

### Issue: Cannot find old image to backup
**Solution**: This is OK if it's your first build. The script will just skip the backup step.

### Issue: Build fails with error
**Solution**: 
```bash
# Clean Docker cache and try again
docker system prune -a
docker-compose build --no-cache
```

### Issue: Want to completely remove a version
```bash
# List images with IDs
docker images | grep pump-fault-risk

# Remove specific version (use IMAGE ID from above)
docker rmi pump-fault-risk-service-app:v1.0.0-before-fixes

# Or remove by ID
docker rmi abc123def456
```

### Issue: Not sure which version is running
```bash
# Check container details
docker ps

# Check logs for model checksum
docker-compose logs | grep "checksum"

# Old version: checksum will differ from new version
# New version: will show PyMuPDF logs
```

---

## Cleanup Old Versions (After Confirming New Works)

Once you've tested the new version and confirm it works:

```bash
# Remove old version image (saves disk space)
docker rmi pump-fault-risk-service-app:v1.0.0-before-fixes

# Optional: Clean up unused images and containers
docker system prune -a
```

**Warning**: Only do this after thoroughly testing the new version!

---

## Quick Reference Card

```bash
# BACKUP CURRENT VERSION
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.0-backup

# BUILD NEW VERSION
docker-compose down
docker-compose build --no-cache
docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:v1.0.1-new
docker-compose up -d

# ROLLBACK TO OLD
docker-compose down
docker tag pump-fault-risk-service-app:v1.0.0-backup pump-fault-risk-service-app:latest
docker-compose up -d

# SWITCH TO NEW
docker-compose down
docker tag pump-fault-risk-service-app:v1.0.1-new pump-fault-risk-service-app:latest
docker-compose up -d

# LIST VERSIONS
docker images | grep pump-fault-risk

# CLEANUP
docker rmi pump-fault-risk-service-app:v1.0.0-backup
```

---

## Best Practices

1. ✅ **Always tag before building** - Create backup before changes
2. ✅ **Test thoroughly** - Run verification script in both environments
3. ✅ **Keep notes** - Document which version is in production
4. ✅ **Use semantic versioning** - e.g., v1.0.0, v1.0.1, v1.1.0
5. ✅ **Clean up periodically** - Remove old versions after confirming new ones work
6. ⚠️ **Don't delete backups immediately** - Keep at least one working version

---

## Production Deployment Workflow

1. **Tag current production image**
   ```bash
   docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:prod-backup-$(date +%Y%m%d)
   ```

2. **Build and test new version locally**
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   # Run tests...
   ```

3. **If tests pass, tag as production-ready**
   ```bash
   docker tag pump-fault-risk-service-app:latest pump-fault-risk-service-app:prod-v1.0.1
   ```

4. **Deploy to production**
   ```bash
   # In production environment
   docker-compose down
   docker-compose up -d
   ```

5. **Monitor for 24-48 hours before removing old backup**
