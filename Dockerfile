# =============================================================================
# Dockerfile -- container runtime for robust-transport-nav (NOT the ARC path)
# =============================================================================
# VT ARC has no Docker: the primary route is arc/setup_env.sh (pip Isaac Sim in
# a conda env). This file is for a Docker-capable GPU box, or as the source image
# for the Apptainer fallback (setup_notes.md Part 1.3).
#
# Built on NVIDIA's Isaac Sim container, which already contains Isaac Sim, its
# bundled Python, and a matching torch build. Isaac Lab is layered on top.
#
# BUILD:
#   docker build -t robust-transport-nav .
#
# RUN (Docker-capable GPU host):
#   docker run --name transport-nav --entrypoint bash -it --runtime=nvidia --gpus all \
#     -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
#     -v $(pwd)/results:/workspace/robust-transport-nav/results:rw \
#     -v ~/.cache/wandb:/root/.cache/wandb:rw \
#     robust-transport-nav
#
# VERIFY (before a long run):
#   1. The Isaac Sim base image tag must match the Isaac Lab version below.
#      A mismatched pair fails at import with confusing extension errors.
#   2. NGC login may be required to pull the base image:
#        docker login nvcr.io
#   3. Mount `results/` as a volume (above) or every artifact dies with the
#      container.
# =============================================================================

# Isaac Sim 4.5.0 <-> Isaac Lab v2.1.0: the code imports the 2.x namespace
# (isaaclab.*). Isaac Lab 1.x (omni.isaac.lab.*) will fail at import.
# VERIFY: pin to the Isaac Sim release your Isaac Lab checkout targets.
FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    ISAACSIM_PATH=/isaac-sim \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        curl \
        vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Isaac Lab
# ---------------------------------------------------------------------------
# VERIFY ON A100: pin a TAG or COMMIT rather than tracking main. Isaac Lab's API
# moves fast, and this repo's `# VERIFY ON A100:` notes are written against a
# specific generation of it. An unpinned clone means the codebase can break
# between two builds with no local change.
ARG ISAACLAB_REF=v2.1.0
RUN git clone https://github.com/isaac-sim/IsaacLab.git isaaclab \
    && cd isaaclab \
    && git checkout ${ISAACLAB_REF}

# Isaac Lab's installer wires its extensions into Isaac Sim's bundled Python.
RUN cd /workspace/isaaclab \
    && ln -sf ${ISAACSIM_PATH} _isaac_sim \
    && ./isaaclab.sh --install rsl_rl

# ---------------------------------------------------------------------------
# Project dependencies
# ---------------------------------------------------------------------------
# Installed into ISAAC SIM'S python. Using system pip here would put the
# packages in an interpreter that never runs this code -- a silent no-op whose
# symptom is an ImportError at training time.
COPY requirements.txt /workspace/robust-transport-nav/requirements.txt
RUN ${ISAACSIM_PATH}/python.sh -m pip install --no-cache-dir \
        -r /workspace/robust-transport-nav/requirements.txt

# ---------------------------------------------------------------------------
# Project source (copied last so code edits do not bust the dependency layers)
# ---------------------------------------------------------------------------
COPY . /workspace/robust-transport-nav
WORKDIR /workspace/robust-transport-nav

ENV PYTHONPATH=/workspace/robust-transport-nav:${PYTHONPATH}

# Fail the BUILD if the pure-logic tests fail. They need no GPU and no Isaac, so
# a broken train/OOD split or a broken reward can never reach the A100 inside a
# green image.
RUN ${ISAACSIM_PATH}/python.sh -m pytest tests/ -q

# Interactive by default: training is launched explicitly, never on container
# start, so a stray `docker run` cannot burn A100 hours.
CMD ["/bin/bash"]
