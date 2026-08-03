# Portable training and multi-GPU launch

All source-configured paths are relative to the repository root. Runtime output,
render caches and temporary IPC files stay under `outputs/` and `runtime/`.
The resolved configuration intentionally records absolute paths for provenance.

Large datasets may either be copied into `data/` or mounted elsewhere. Copy the
versioned example into the ignored machine-local configuration, then edit it:

```powershell
Copy-Item configs/local_paths.example.yaml configs/local_paths.yaml
notepad configs/local_paths.yaml
```

The same YAML file is used on Linux. Paths can also be supplied explicitly to
`training.py`; project dataset paths do not depend on environment
variables. The accurate FR5/AG-160-95 URDF and meshes are bundled under
`assets/robots/FR5_AG-160-95`.

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
without padding, training groups are weighted and sharded, and checkpoint files
are written only by rank 0 while preserving per-rank RNG state. Gradient
accumulation remains available as an explicit experiment override, but the
formal default is 1 and therefore performs no accumulation.
Training losses are reduced across ranks before rank 0 writes terminal, JSONL
and TensorBoard metrics; count metrics are summed rather than averaged.
