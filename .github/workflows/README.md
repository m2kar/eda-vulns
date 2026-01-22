# GitHub Actions Workflows

This directory contains automated workflows for the eda-vulns repository.

## Available Workflows

### 1. Docker Build and Push (`docker-build.yml`)

Automatically builds Docker images and pushes them to GitHub Container Registry (ghcr.io).

**Triggers:**
- Push to `main` branch (when Dockerfile or related files change)
- Pull requests to `main` branch
- Manual workflow dispatch

**What it does:**
- Builds Docker image for `circt-1` vulnerability
- Pushes to `ghcr.io/m2kar/eda-vulns/circt-1`
- Creates multiple tags:
  - `latest` (for main branch)
  - `firtool-1.139.0` (specific version)
  - `main-<sha>` (commit-based)
  - Branch/PR tags
- Uses GitHub Actions cache for faster builds
- Generates build attestation for supply chain security

**Platforms:** linux/amd64

**Pull the image:**
```bash
docker pull ghcr.io/m2kar/eda-vulns/circt-1:latest
```

### 2. Docker Test (`docker-test.yml`)

Runs automated tests to verify Docker images work correctly.

**Triggers:**
- Push to `main` branch (when circt-1 files change)
- Pull requests to `main` branch
- Manual workflow dispatch

**What it does:**
- Builds Docker image from source
- Runs full vulnerability reproduction test
- Tests vulnerable code only
- Tests workaround code only
- Extracts and uploads test results as artifacts

**Test artifacts:**
- Available for 30 days after workflow run
- Includes error outputs, IR dumps, and compilation results

## Usage

### Running Workflows Manually

1. Go to the "Actions" tab in GitHub repository
2. Select the workflow you want to run
3. Click "Run workflow" button
4. Choose the branch and click "Run workflow"

### Using the Docker Images

#### Pull from GitHub Container Registry:

```bash
# Pull latest version
docker pull ghcr.io/m2kar/eda-vulns/circt-1:latest

# Pull specific version
docker pull ghcr.io/m2kar/eda-vulns/circt-1:firtool-1.139.0

# Run vulnerability reproduction
docker run --platform linux/amd64 --rm \
  ghcr.io/m2kar/eda-vulns/circt-1:latest
```

#### Save test results:

```bash
docker run --platform linux/amd64 --rm \
  -v $(pwd)/results:/vuln-reproduction/output \
  ghcr.io/m2kar/eda-vulns/circt-1:latest
```

## Permissions

The workflows require the following permissions:
- `contents: read` - Read repository contents
- `packages: write` - Push to GitHub Container Registry

These are automatically provided by GitHub Actions.

## Security

- Images are built from trusted base: `ubuntu:24.04`
- Build provenance attestation is generated
- No secrets are required (uses `GITHUB_TOKEN`)
- All builds are reproducible and cached

## Monitoring

Check workflow status:
- Repository → Actions tab
- Green checkmark: Success
- Red X: Failed
- Yellow circle: In progress

View logs:
- Click on any workflow run
- Click on the job name
- Expand steps to see detailed logs

## Troubleshooting

### Build fails on push

1. Check if Dockerfile syntax is correct
2. Verify all COPY paths exist
3. Check workflow logs for specific errors

### Image not appearing in registry

1. Ensure you have `packages: write` permission
2. Check if workflow completed successfully
3. PR builds don't push images (only builds for testing)

### Tests fail

1. Check test-artifacts for detailed error logs
2. Verify Docker image builds successfully locally
3. Review reproduce.sh script for issues

## Adding New Vulnerabilities

When adding a new vulnerability directory (e.g., `yosys-1/`):

1. Create similar workflow file or update existing one
2. Add new job for the vulnerability
3. Update image paths and names
4. Test locally before committing

Example for new tool:
```yaml
build-yosys-1:
  runs-on: ubuntu-latest
  # ... similar steps as circt-1
```

## Related Documentation

- [Docker Build Action](https://github.com/docker/build-push-action)
- [Docker Metadata Action](https://github.com/docker/metadata-action)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
