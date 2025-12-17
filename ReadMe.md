# 3DGRUT Reconstruction Pipeline

This document describes the **standard operating procedure** for generating a 3DGRUT splat from a video sequence.

Please follow the steps **in order**. Skipping steps or modifying inputs without guidance may result in failed reconstructions.

---

## Setup

The setup section covers **all one-time configuration** required before running any reconstructions.

### Prerequisites

Before starting, ensure the following:

* Linux workstation (x86_64)
* NVIDIA GPU with a working driver (`nvidia-smi` should run)
* Docker installed and working
* Docker Compose plugin available (`docker compose`)

If any of the above are missing, contact the project maintainer before proceeding.

---

### One-Time Environment Setup

All required setup (user IDs, environment variables, Docker configuration, and image builds) is handled automatically using the provided setup script.

Run the following **once per machine or fresh clone**:

```bash
bash scripts/setup/build.sh
```

This step:

* Generates the required `.env` file
* Builds all necessary Docker images
* Configures the environment for downstream pipelines

**No manual Docker commands are required.**

---

### Hugging Face Token (Required for SAM3)

Some pipelines (e.g., **SAM3 segmentation**) require access to gated Hugging Face models. This requires a valid Hugging Face access token and prior model approval.

#### 1. Create a Hugging Face Account

If you do not already have one, create an account at:

```
https://huggingface.co/join
```

#### 2. Request Access to SAM3

SAM3 is a gated model and requires explicit access approval.

1. Visit the SAM3 model page on Hugging Face
2. Click **Request Access**
3. Wait for approval (this may take several hours to a day)

You will not be able to download or use SAM3 until access is granted.

#### 3. Generate a Hugging Face Token

Once access is approved:

1. Go to **Settings → Access Tokens** on Hugging Face
2. Create a **Read** token
3. Copy the token (it will only be shown once)

#### 4. Add the Token to `.env`

Open the generated `.env` file and add the following line:

```text
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Do not commit this token to Git.**

The `.env` file is already listed in `.gitignore` and is loaded automatically by Docker Compose.

---

## Workflow

This section describes the **end-to-end reconstruction workflow**. These steps are repeated **once per video** after setup is complete.

---

### Step 1: Extract Frames from the Video

Extract image frames from the input video using FFmpeg:

```bash
bash scripts/extract_frames.sh <VIDEO_PATH> <FRAME_STRIDE> <VIDEO_DIR>/ffmpeg_frames
```

**Arguments**

* `<VIDEO_PATH>`: Path to the input video file
* `<FRAME_STRIDE>`: Number of frames to skip between extractions
* `<VIDEO_DIR>`: Directory used to store all outputs for this video

A smaller frame stride produces more frames and higher-quality reconstructions, but increases runtime.

---

### Step 2: Audit Extracted Frames (Manual Step)

Open the extracted frames and visually inspect them:

* Remove blurry or motion-distorted frames
* Ensure sufficient coverage of the plant from multiple viewpoints
* If too few usable frames are available, re-run extraction with a smaller frame stride

**This step is critical for reconstruction quality and should not be skipped.**

---

### Step 3: Run COLMAP Reconstruction

Run the COLMAP pipeline on the extracted frames:

```bash
bash scripts/colmap.sh <VIDEO_DIR>/ffmpeg_frames <VIDEO_DIR>/colmap
```

This step generates camera poses and sparse geometry. The output directory will contain multiple stages, including `stage2`, which is required for the next step.

---

### Step 4: Run 3DGRUT Splat Generation

Generate the 3DGRUT splat using the refined COLMAP output:

```bash
bash scripts/3dgrut.sh <VIDEO_DIR>/colmap/stage2 <VIDEO_DIR>/3dgrut
```

This step is compute-intensive and typically takes **approximately 3 hours**. Do not interrupt the process once it has started.

---

### Step 5: Post-Process and Edit the Splat

1. Launch the SuperSplat editor container:

   ```bash
   bash scripts/supersplat.sh
   ```

2. In a browser, open: `https://localhost:3001`

3. Navigate to **File → Import**, then import the generated `.ply` file from the previous step.

4. Use the editor tools to remove non-plant points and artifacts so that the splat contains **only the plant geometry**.

5. Export the cleaned splat as:

   **<VIDEO_DIR>/edited_gauss.ply**

---

## Summary

```text
setup/build.sh (one-time setup)
        ↓
extract_frames.sh
        ↓
manual frame audit
        ↓
colmap.sh
        ↓
3dgrut.sh
        ↓
supersplat editing
```

---

## Notes for Students

* Run the setup script before doing anything else
* Do not manually edit Docker files or environment variables unless instructed
* Keep all outputs organized within the specified `VIDEO_DIR`
* If a step fails, do not attempt to skip ahead—debug or ask for help
* Record any issues or observations for discussion with the research team

---

## Support

If you encounter errors or unexpected behavior, document:

* The command used
* The full error message
* Your system (GPU type, OS)

Then contact the project maintainer for assistance.
