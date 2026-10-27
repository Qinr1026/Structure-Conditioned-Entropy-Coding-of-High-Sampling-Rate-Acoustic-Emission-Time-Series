"""Generic training on caller-supplied blocks; no experiment-specific loaders.

Train the neural context first, then select structural-offset strength only
on validation blocks. This configurable recipe is not the paper's archived
multi-stage training run and does not reproduce its metrics without its data.
"""
import copy
import numpy as np
import torch
from .model import StateContext
from .priors import fit
from .quantization import symbols


def _blocks(values, label):
    x = np.asarray(values)
    if x.ndim != 2 or min(x.shape) < 1 or x.shape[1] % 8:
        raise ValueError(label+' must have shape (blocks, samples), with samples divisible by eight')
    if not np.isfinite(x).all() or x.min() < -32768 or x.max() > 32767 or not np.array_equal(x, np.rint(x)):
        raise ValueError(label+' must contain finite signed 16-bit ADC values')
    return x


def _quantized(values, steps):
    pairs = [[symbols(x, float(step), 'e8') for x in values] for step in steps]
    q = torch.from_numpy(np.stack([np.stack([v[0] for v in row]) for row in pairs]))
    c = torch.from_numpy(np.stack([np.stack([v[1] for v in row]) for row in pairs]))
    return q, c


@torch.no_grad()
def _score(model, q, c, length, batch_size):
    device = next(model.parameters()).device
    model.eval()
    rates = []
    for pi in range(len(model.steps)):
        for start in range(0, q.shape[1], batch_size):
            qq = q[pi, start:start+batch_size].to(device)
            cc = c[pi, start:start+batch_size].to(device)
            profile = torch.full((len(qq),), pi, device=device, dtype=torch.long)
            rates.extend((model.rate(qq, cc, profile)/length).cpu().tolist())
    return float(np.mean(rates))


def train_model(train_blocks, validation_blocks, steps, *, bands=64, epochs=12,
                batch_size=8, learning_rate=2e-4, seed=47, device='cpu',
                strengths=(0., .25, .5, .75, 1.)):
    """Return a trained model and scalar training/selection history.

    The caller must split by acquisition condition before making blocks.
    Test samples must never be passed to either input.
    """
    train = _blocks(train_blocks, 'training input')
    val = _blocks(validation_blocks, 'validation input')
    steps = np.asarray(steps, dtype=np.float32)
    if train.shape[1] != val.shape[1]:
        raise ValueError('training and validation block lengths must match')
    if steps.ndim != 1 or not 1 <= len(steps) <= 256 or not np.isfinite(steps).all() or (steps <= 0).any() or len(set(steps.tolist())) != len(steps):
        raise ValueError('steps must be distinct positive finite values (at most 256)')
    if bands < 1 or bands > 65535 or train.shape[1] % (8*bands):
        raise ValueError('block length must be divisible by eight times bands')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('invalid training budget')
    if not strengths or not all(np.isfinite(a) and 0 <= a <= 1 for a in strengths):
        raise ValueError('strengths must be a nonempty sequence in [0, 1]')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    prior = fit(train, steps, 'e8', bands=bands)
    model = StateContext(prior, train.shape[1], 'e8', True, False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    qt, ct = _quantized(train, steps)
    qv, cv = _quantized(val, steps)
    best_score = float('inf')
    best_state = None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            profiles = rng.integers(0, len(steps), len(ids))
            q = qt[profiles, ids].to(device)
            c = ct[profiles, ids].to(device)
            pi = torch.as_tensor(profiles, device=device)
            opt.zero_grad(set_to_none=True)
            rate = model.rate(q, c, pi).mean()/train.shape[1]
            loss = rate + 1e-5*sum(p.square().mean() for p in model.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            losses.append(float(rate.detach()))
        score = _score(model, qv, cv, val.shape[1], batch_size)
        history.append(dict(epoch=epoch, train_estimated_bps=float(np.mean(losses)), validation_estimated_bps=score))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('no finite validation checkpoint was obtained')
    model.load_state_dict(best_state)
    original_offsets = {k: getattr(model, 'state_'+k+'_offset').detach().clone() for k in ('support','scale','sign')}
    model.conditioned = True
    selection = []
    for alpha in strengths:
        for key, value in original_offsets.items():
            getattr(model, 'state_'+key+'_offset').copy_(value*alpha)
        selection.append(dict(alpha=float(alpha), validation_estimated_bps=_score(model, qv, cv, val.shape[1], batch_size)))
    chosen = min(selection, key=lambda row: row['validation_estimated_bps'])['alpha']
    for key, value in original_offsets.items():
        getattr(model, 'state_'+key+'_offset').copy_(value*chosen)
    return model.eval(), dict(seed=seed, training=history, strength_selection=selection, selected_alpha=chosen)
