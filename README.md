# Pix2Video

## Install

We use ffmpeg to create short clips of results
```bash
# ffmpeg
sudo apt install ffmpeg
```

Create a clean conda environment:
```bash
# Create clean conda environment
conda create -y -n pix2video
conda activate pix2video
```

Install a compatible Pytorch version with your system, e.g.:
```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install required packages. First of all, we make changes in the `diffusers` library code so install the custom version:
```bash
cd pix2video
# Install custom diffusers library
pip install -e mydiffusers
# Install other required packages
pip install -r requirements.txt
```

Get huggingface token required to download StableDiffusion models and add it to `test_cfg.py` at line 36.

You can then run the code as follows. The options in the `test_cfg.py` provide explanations. We also provide an example data sequence to try.

```bash
python test_cfg.py
```

## Citations

If you make use of the work, please cite our paper.

```bibtex
@inproceedings{ceylan2023pix2video,
  title={Pix2Video: Video Editing using Image Diffusion},
  author={Ceylan, Duygu and Huang, Chun-Hao Paul and Mitra, Niloy J},
  conference={ICCV},
  year={2023}
}
}
```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!

## License
This code is released under the Adobe Research License agreement and is for non-commercial and research purposes only.

## Contact

If you have any questions, feel free to open an issue or contact us through e-mail (duygu.ceylan@gmail.com or chunhaoh@adobe.com).
