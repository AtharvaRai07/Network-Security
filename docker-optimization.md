# Docker and GitHub Actions Workflow Optimization

This file provides information on optimizing the Docker build and GitHub Actions workflow to handle disk space issues.

## Disk Space Issues

When running into "no space left on device" errors in GitHub Actions, consider the following strategies:

### 1. Multi-stage Docker Builds

The Dockerfile has been optimized with a multi-stage build to reduce the final image size:
- First stage installs build dependencies and Python packages
- Second stage copies only the necessary files from the first stage

### 2. GitHub Actions Cleanup Steps

Add these steps to your GitHub Actions workflow to free up disk space:

```yaml
- name: Free disk space
  run: |
    # Remove unnecessary large packages
    sudo apt-get remove --purge -y azure-cli ghc* zulu* hhvm llvm* firefox google* dotnet* powershell openjdk* mysql* php* android*
    sudo apt-get autoremove -y
    sudo apt-get clean

    # Remove large directories
    sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc "$AGENT_TOOLSDIRECTORY"

    # Show available disk space
    df -h
```

### 3. Selective Package Installation

- Install only required packages
- Pin versions to avoid unexpected upgrades
- Use `--no-cache-dir` with pip installations

### 4. Docker Pull Optimization

For large images like this one with machine learning libraries, consider:
- Setting up Docker layer caching
- Pulling from a closer registry
- Breaking dependencies into smaller, more manageable containers
