# Portable training and multi-GPU launch

All source-configured paths are relative to the repository root. Runtime output,
render caches and temporary IPC files stay under `outputs/` and `runtime/`.
The resolved configuration intentionally records absolute paths for provenance.

Large datasets may either be copied into `data/` or mounted elsewhere. Override
the defaults without editing source files:

```powershell
$env:TCD_DATASET_ROOT = "E:\datasets\TaskOrientedClutterSceneDataset"
$env:TCD_ACRONYM_ROOT = "E:\datasets\ACRONYM"
$env:TCD_FUNCTIONAL_REGION_ROOT = "E:\datasets\manual_function_regions_v1"
$env:TCD_PYBULLET_PYTHON = "D:\Anaconda\envs\gapg\python.exe"
```

Linux uses the same variable names. The accurate FR5/AG-160-95 URDF and meshes
are bundled under `assets/robots/FR5_AG-160-95`.

Initialize the pinned official PTv3 source once after cloning:

```powershell
git submodule update --init third_party/PointTransformerV3
```

The checked-in Windows environment uses PyTorch/CUDA 11.8,
`spconv-cu118`, PyG and `backbone.enable_flash_attention=false`. This is the
primary supported desktop path. Linux servers may use a matching CUDA-12
PyTorch/spconv build plus FlashAttention and override:

```bash
python -m tcd_prg.scripts.train --config configs/config.yaml \
  backbone.enable_flash_attention=true
```

Do not compile a CUDA-12 FlashAttention extension into a PyTorch CUDA-11.8
environment; create a version-aligned server or dedicated Windows environment.

Windows multi-GPU training:

```powershell
.\scripts\train_ddp.ps1 -Gpus 2
```

Linux multi-GPU training:

```bash
bash scripts/train_ddp.sh 2
```

Both launchers accept ordinary dot-list overrides after their GPU-count
argument. Windows uses Gloo with a repository-local `file://` rendezvous so it
does not depend on host-name/DNS behavior; Linux CUDA defaults to NCCL through
`torchrun`. Each process owns one GPU. Validation is sharded
without padding, training groups are weighted and sharded, gradient accumulation
uses `no_sync`, and checkpoint files are written only by rank 0 while preserving
per-rank RNG state.
