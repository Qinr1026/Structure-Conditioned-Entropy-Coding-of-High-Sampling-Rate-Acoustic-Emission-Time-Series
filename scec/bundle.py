"""Read/write a caller-owned model bundle. No pretrained models are distributed."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .model import StateContext


def save_model(model, folder, metadata=None):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('choose an empty model output directory')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'model.pt'
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({'model': state}, path)
    config = dict(method_name='Structure-Conditioned Entropy Coding',
                  block_size=model.block_size, hidden=model.input_projection.out_channels,
                  quantizer=model.quantizer, neural=model.neural, conditioned=model.conditioned,
                  resonance_centers_normalized=list(model.resonance_centers_normalized),
                  model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  stream_magic='SCE8', stream_version=1, header_bytes=37,
                  metadata=metadata or {})
    (folder/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config


def load_model(folder, device='cpu'):
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    path = folder/'model.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['model_sha256']:
        raise ValueError('model checksum mismatch')
    sd = torch.load(path, map_location='cpu', weights_only=True)['model']
    prior = {'steps': sd['steps'].numpy(), 'coset': sd['base_coset'].numpy()}
    for key in ('support', 'scale', 'sign'):
        pooled = sd['base_'+key].numpy()[:, :, None, :]
        prior['pooled_'+key] = pooled
        prior[key] = np.repeat(pooled, 4, axis=2)
    model = StateContext(prior, config['block_size'], config.get('quantizer', 'e8'),
                         config.get('neural', True), config.get('conditioned', True),
                         config.get('hidden', 48))
    if 'resonance_centers_normalized' in config:
        centers = tuple(config['resonance_centers_normalized'])
        if len(centers) != 5 or not all(np.isfinite(v) and 0 <= v <= 1 for v in centers):
            raise ValueError('invalid normalized frequency-feature centers')
        model.resonance_centers_normalized = centers
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device), config
