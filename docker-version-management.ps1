# Docker Version Management Script (PowerShell)
# Allows building new versions while keeping old ones for rollback

$ErrorActionPreference = "Stop"

# Version variables
$OLD_VERSION = "v1.0.0-before-fixes"
$NEW_VERSION = "v1.0.1-with-pdf-fixes"
$IMAGE_NAME = "pump-fault-risk-service-app"  # From docker-compose.yml

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Docker Version Management" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Tag current running image as old version (backup)
Write-Host ""
Write-Host "Step 1: Backing up current Docker image..." -ForegroundColor Yellow

$currentImage = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String -Pattern "^${IMAGE_NAME}:latest$"

if ($currentImage) {
    docker tag "${IMAGE_NAME}:latest" "${IMAGE_NAME}:${OLD_VERSION}"
    Write-Host "✓ Tagged current image as: ${IMAGE_NAME}:${OLD_VERSION}" -ForegroundColor Green
} else {
    Write-Host "⚠ No current image found to backup (this is OK for first build)" -ForegroundColor Yellow
}

# Step 2: Build new image with fixes
Write-Host ""
Write-Host "Step 2: Building new Docker image with fixes..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray
docker-compose build --no-cache

# Step 3: Tag new image with new version
Write-Host ""
Write-Host "Step 3: Tagging new image..." -ForegroundColor Yellow
docker tag "${IMAGE_NAME}:latest" "${IMAGE_NAME}:${NEW_VERSION}"
Write-Host "✓ Tagged new image as: ${IMAGE_NAME}:${NEW_VERSION}" -ForegroundColor Green

# Step 4: Show available versions
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Available Docker Image Versions:" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
docker images | Select-String -Pattern $IMAGE_NAME

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To use the NEW version (recommended):" -ForegroundColor Cyan
Write-Host "  docker-compose down" -ForegroundColor White
Write-Host "  docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "To rollback to OLD version:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor White
Write-Host "  docker tag ${IMAGE_NAME}:${OLD_VERSION} ${IMAGE_NAME}:latest" -ForegroundColor White
Write-Host "  docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "To switch back to NEW version:" -ForegroundColor Cyan
Write-Host "  docker-compose down" -ForegroundColor White
Write-Host "  docker tag ${IMAGE_NAME}:${NEW_VERSION} ${IMAGE_NAME}:latest" -ForegroundColor White
Write-Host "  docker-compose up" -ForegroundColor White
Write-Host ""
