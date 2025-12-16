# 3DGRUT Reconstruction Pipeline

This document describes the **standard operating procedure** for generating a 3DGRUT splat from a video sequence.

Please follow the steps **in order**. Skipping steps or modifying inputs without guidance may result in failed reconstructions.

---

## Prerequisites

Before starting, ensure the following:

* Linux workstation (x86_64 or Jetson)
* NVIDIA GPU with a working driver (`nvidia-smi` should run)
* Docker installed and working
* Docker Compose plugin available (`docker compose`)
* `ffmpeg` installed on the host system

If any of the above are missing, contact the project maintainer before proceeding.

---

## Step 1: Prepare the Repository and Submodules

Initialize and prepare all required submodules (including 3DGRUT dependencies):

```bash
bash scripts/prepare_3dgrut.sh
```

This step only needs to be run once per fresh clone or after repository updates.

---

## Step 2: Build Docker Images

Build all Docker images required for the pipeline:

```bash
docker compose -f docker/compose/docker-compose.yml build
```

This may take several minutes depending on your system and network speed.

---

## Step 3: Extract Frames from the Video

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

## Step 4: Audit Extracted Frames (Manual Step)

Open the extracted frames and visually inspect them:

* Remove blurry or motion-distorted frames
* Ensure sufficient coverage of the plant from multiple viewpoints
* If too few usable frames are available, re-run extraction with a smaller frame stride

**This step is critical for reconstruction quality and should not be skipped.**

---

## Step 5: Run COLMAP Reconstruction

Run the COLMAP pipeline on the extracted frames:

```bash
bash scripts/colmap.sh <VIDEO_DIR>/ffmpeg_frames <VIDEO_DIR>/colmap
```

This step generates camera poses and sparse geometry. The output directory will contain multiple stages, including `stage2`, which is required for the next step.

---

## Step 6: Run 3DGRUT Splat Generation

Generate the 3DGRUT splat using the refined COLMAP output:

```bash
bash scripts/3dgrut.sh <VIDEO_DIR>/colmap/stage2 <VIDEO_DIR>/3dgrut
```

This step is compute-intensive and typically takes **approximately 2 hours**. Do not interrupt the process once it has started.

---

## Step 7: Post-Processing and Editing

1. Navigate to **[https://supersplat.com](https://supersplat.com)**
2. Upload the generated splat from:

   ```
   <VIDEO_DIR>/3dgrut
   ```
3. Edit the splat to retain **only the plant geometry**, removing background and artifacts

---

## Summary

```text
prepare_3dgrut.sh
        ↓
docker compose build
        ↓
extract_frames.sh
        ↓
manual frame audit
        ↓
colmap.sh
        ↓
3dgrut.sh
        ↓
supersplat.com editing
```

---

## Notes for Students

* Follow the instructions exactly as written
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
