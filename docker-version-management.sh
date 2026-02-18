#!/bin/bash
# Docker Version Management Script
# Allows building new versions while keeping old ones for rollback

set -e

# Version variables
OLD_VERSION="v1.0.0-before-fixes"
NEW_VERSION="v1.0.1-with-pdf-fixes"
IMAGE_NAME="pump-fault-risk"

echo "=========================================="
echo "Docker Version Management"
echo "=========================================="

# Step 1: Tag current running image as old version (backup)
echo ""
echo "Step 1: Backing up current Docker image..."
if docker images | grep -q "$IMAGE_NAME.*latest"; then
    docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${OLD_VERSION}
    echo "✓ Tagged current image as: ${IMAGE_NAME}:${OLD_VERSION}"
else
    echo "⚠ No current image found to backup (this is OK for first build)"
fi

# Step 2: Build new image with fixes
echo ""
echo "Step 2: Building new Docker image with fixes..."
docker-compose build --no-cache

# Step 3: Tag new image with new version
echo ""
echo "Step 3: Tagging new image..."
docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${NEW_VERSION}
echo "✓ Tagged new image as: ${IMAGE_NAME}:${NEW_VERSION}"

# Step 4: Show available versions
echo ""
echo "=========================================="
echo "Available Docker Image Versions:"
echo "=========================================="
docker images | grep "$IMAGE_NAME" || echo "No images found"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "To use the NEW version (recommended):"
echo "  docker-compose up"
echo ""
echo "To rollback to OLD version:"
echo "  docker tag ${IMAGE_NAME}:${OLD_VERSION} ${IMAGE_NAME}:latest"
echo "  docker-compose up"
echo ""
echo "To switch back to NEW version:"
echo "  docker tag ${IMAGE_NAME}:${NEW_VERSION} ${IMAGE_NAME}:latest"
echo "  docker-compose up"
echo ""
