import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
from pathlib import Path

import click


@click.command(help='Structure-from-Motion from MoGe-3 metric geometry. Recovers camera '
                    'poses for a folder of images and writes a COLMAP model to <output>/sparse/0/.')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True, file_okay=False),
              required=True, help='Folder of input images ("jpg"/"png").')
@click.option('--output', '-o', 'output_path', type=click.Path(), default='./output',
              help='Output folder; the COLMAP model is written to <output>/sparse/0/.')
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), default=None,
              help='YAML config; its `sfm:` block populates MoGe3SfMConfig. Omit for defaults.')
@click.option('--pretrained', 'pretrained', type=str, default=None,
              help='MoGe-3 checkpoint (HF id or local .pt). Overrides the config.')
@click.option('--device', 'device', type=str, default=None,
              help='Device, e.g. "cuda" / "cpu". Overrides the config.')
def main(input_path, output_path, config_path, pretrained, device):
    from moge.sfm.pipeline import load_config, run_moge3_sfm

    cfg = load_config(Path(config_path) if config_path else None)
    if pretrained is not None:
        cfg.moge_pretrained = pretrained
    if device is not None:
        cfg.device = device
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    run_moge3_sfm(Path(input_path), out, cfg)


if __name__ == '__main__':
    main()
