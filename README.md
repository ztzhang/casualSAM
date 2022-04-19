## casualSAM
### non-exhaustive requirements:
2. skimage
3. ImageIO
4. Pillow
5. configargparse
6. tqdm

### download midas checkpoints:
`bash download_depth_ckpt.sh`

### fill in davis dataset path
L12 in configs/__init__.py

### run training
`bash ./experiments/davis/train.sh`
