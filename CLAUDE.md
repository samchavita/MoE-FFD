# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official PyTorch implementation of **MoE-FFD** (Mixture of Experts for generalized and parameter-efficient Face Forgery Detection, IEEE TDSC 2025). A frozen ImageNet-21k ViT-B/16 is fine-tuned for binary real/fake classification by training only small MoE modules injected into each transformer block. Pre-trained weights: https://huggingface.co/luobo91/MoE-FFD/tree/main

## Environment

- Use the `moeffd` conda env (`/home/user/anaconda3/envs/moeffd/bin/python`). The base anaconda Python lacks `cv2`/`torch`.
- Versions installed in the env are `torch==2.2.2+cu121`, `torchvision==0.17.2`, `timm==0.5.4` and `albumentations==1.1.0`, and train/eval run on them. `timm==0.5.4` is the one that matters, because ViT_MoE.py imports timm internals like `timm.models.helpers.build_model_with_cfg` that moved in later versions. `requirements.txt` still pins `torch==1.8.0`, which is not what's installed.
- Training and evaluation assume CUDA (`.cuda()` calls are hard-coded throughout).

## Commands

There is no test suite, linter, or build step.

```bash
python train.py [--device 0 --model_dir models/train --epochs 20 -bs 32 -lr 3e-5 --resume <epoch>]
python train.py -bs 16 --accum_steps 2   # effective batch 32 on an 8 GB GPU (bs 32 runs out of memory in fp32)
python eval.py --model_path models/train/models_params_<epoch>.tar   # or models/train/model_params_best_*.pkl
python make_sample_data.py --out sample_data      # tiny synthetic FF++-layout dataset for smoke tests
python faceforensics_download_v4.py <output_path> -d all -c c23 -t videos
python extract_faces.py --src <FF++ root> --dst <out> --splits splits   # videos -> 224px face crops in <out>/{train,valid,test}/<method>/<id>/
python extract.py --data_path <FF++ root> -d all -c c23   # all full frames, written next to the videos (not the split layout)
```

**Dataset paths are hard-coded**, not CLI args: `train.py` uses `/data3/law/data/FF++/c23/{train,valid}` and `eval.py` uses `/data3/law/data/Celeb_DF/test` with `dataset='CDF'`. Edit these in the `__main__` block to run on other data.

## Architecture

**Model (`ViT_MoE.py`)**: a modified copy of timm's `vision_transformer.py`. Always build it with `vit_base_patch16_224_in21k(pretrained=True, num_classes=2)`. Two MoE components are added to every `Block`:
- `LoRA_MoElayer` (inside `Attention`): 7 LoRA experts with ranks `[8,16,32,48,64,96,128]` that produce a delta added to q/k/v. Controlled by `lora_topk`.
- `Adapter_MoElayer` (in parallel with the MLP): 5 `Conv2d_Adapter` experts, each a bottleneck (dim 8) around a `Conv2d_Diff` 3×3 conv over the 14×14 patch grid. The variants are `cv` vanilla, `cd` center-difference, `ad` angular-difference, `rd` radial-difference and `scd`. Controlled by `adapter_topk`.
- Both use noisy top-k gating (`w_gate`, `w_noise`) driven by the token-averaged feature, plus a `SparseDispatcher`. Each returns a load-balancing loss (`cv_squared` of importance/load).
- `VisionTransformer.freeze_stages()` runs in `__init__` and sets `requires_grad=False` on every parameter whose name lacks `LoRA`, `adapter`, `head` or `norm1`. Renaming modules or parameters changes what gets trained.
- `forward()` returns `(logits, moe_loss)`, where `moe_loss = 200*mean(lora_loss) + 1*mean(adapter_loss)`. Callers must unpack a tuple.

**`demo.py`** is a near-duplicate of `ViT_MoE.py` whose `forward` returns `(logits, lora_loss, adapter_loss)` (three values). Its `__main__` loads a released checkpoint and shows the inference preprocessing. Changes to the model usually have to be mirrored in both files.

**Data (`dataset.py`)**: the expected layout is `<split>/<method>/<video_id>/<frame>.png` (see the README's Data Structure section).
- Label rule: the `original` folder is real (0) and every other method folder is fake (1). For CDF/DFDC-P, any folder name containing `real` is labeled 0.
- Frame filenames must be integers (sorted by `int(name[:-4])`).
- `FFPP_Dataset` (training): returns single frames, sampling `frame` frames per video evenly. On the train split it samples `4×frame` from `original` to balance classes.
- `TestDataset` (val/test): returns one stacked tensor of `frame` frames per video, shape `(F,3,224,224)`. Loaders use `batch_size=1` and `squeeze(0)`, so the frames act as the batch.
- Preprocessing: resize to 224 and normalize with mean/std 0.5. No augmentation.

**Training loop (`train.py`)**:
- Loss is `CE + moe_loss`.
- `--accum_steps N` accumulates gradients over N batches before each `optimizer.step()`, giving an effective batch of `batch_size × N`. The epoch's last step may use fewer batches. `--record_step` still counts batches, not optimizer steps.
- On an RTX 4060 8 GB, batch 16 peaks at about 5.3 GB, and batch 32 runs out of memory. Throughput is about 51 frames/s, so a full epoch takes about 40 min (115k train frames plus validation) and 20 epochs take about 13–14 h.
- Adam uses two param groups: gating params (`w_gate`/`w_noise`) start at lr 1e-4 and everything else at `--learning_rate`. `StepLR(step=5, gamma=0.5)`.
- Every epoch it saves `models_params_<epoch>.tar` (`model_state_dict` + `optimizer_state_dict`) to `--model_dir` and logs to `<model_dir>/train.log`.
- Validation reports video-level metrics (mean fake prob over frames) and frame-level ACC/AUC/EER via `utils.cal_metrics`. Selection is by video AUC. The best weights are saved as `<model_dir>/model_params_best_<acc>auc<auc>epoch<n>.pkl` (a bare `state_dict`), and the previous best file is deleted. `train()` returns the best video AUC.
- Training is plain fp32, with no AMP.

## Data preparation

- `faceforensics_download_v4.py` writes `<root>/original_sequences/youtube/<comp>/videos/` and `<root>/manipulated_sequences/<Method>/<comp>/videos/`. `extract.py` and `extract_faces.py` both read that layout.
- `extract_faces.py` produces the training layout directly. It uses the official FF++ splits in `splits/{train,val,test}.json`, crops faces with MTCNN (`facenet-pytorch`), and saves only the frames the loader samples: 80 per train original and 20 for every other video.
- `extract.py` writes every frame to `<root>/<method path>/<comp>/images/<video_id>/NNNN.png`. That output is not in the split layout, so it can't be fed to `dataset.py` without being reorganized.
