# CIRCT Vulnerability CVE-XXXX-XXXXX Reproduction Environment
# Vulnerability: Inconsistent Array Indexing Handling in Sensitivity Lists
# Affected Version: CIRCT firtool-1.139.0
# Issue: https://github.com/llvm/circt/issues/9469

FROM ubuntu:24.04

LABEL maintainer="M2kar <m2kar@github>"
LABEL description="Reproduction environment for CIRCT vulnerability - array indexing in sensitivity lists"
LABEL version="1.0"
LABEL circt.version="firtool-1.139.0"
LABEL vulnerability.id="CVE-PENDING"
LABEL vulnerability.cvss="5.3"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install required dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /vuln-reproduction

# Download and extract CIRCT precompiled package (vulnerable version)
RUN wget -q https://github.com/llvm/circt/releases/download/firtool-1.139.0/circt-full-shared-linux-x64.tar.gz \
    && tar -xzf circt-full-shared-linux-x64.tar.gz \
    && rm circt-full-shared-linux-x64.tar.gz

# Add CIRCT binaries to PATH
ENV PATH="/vuln-reproduction/firtool-1.139.0/bin:${PATH}"
ENV LD_LIBRARY_PATH="/vuln-reproduction/firtool-1.139.0/lib:${LD_LIBRARY_PATH}"

# Copy test files
COPY top1.sv /vuln-reproduction/
COPY top2.sv /vuln-reproduction/
COPY reproduce.sh /vuln-reproduction/

# Make reproduction script executable
RUN chmod +x /vuln-reproduction/reproduce.sh

# Verify CIRCT installation
RUN circt-verilog --version && \
    arcilator --version && \
    echo "CIRCT installation successful"

# Create output directory for results
RUN mkdir -p /vuln-reproduction/output

# Set entrypoint to reproduction script
ENTRYPOINT ["/vuln-reproduction/reproduce.sh"]

# Default command: run full reproduction
CMD ["--all"]
