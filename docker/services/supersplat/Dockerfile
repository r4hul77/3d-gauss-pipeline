FROM node:18-bullseye

# SuperSplat requires Node 18+ (per upstream README)
# We'll also install git to clone the repo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Pin to a tag or commit for reproducibility.
# (Example: v2.16.1 is shown as a recent release on GitHub.)
ARG SUPERSPLAT_REF=v2.16.1
RUN git clone https://github.com/playcanvas/supersplat.git \
 && cd supersplat \
 && git checkout "${SUPERSPLAT_REF}"

WORKDIR /opt/supersplat

# Install dependencies (prefer npm ci when package-lock.json exists)
RUN npm ci

EXPOSE 3001

# Default: dev server
CMD ["npm", "run", "develop"]
