"""Train on separately supplied local arrays. No dataset is bundled."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scec.training import train_model
from scec.bundle import save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='local .npy array: training blocks x samples')
    parser.add_argument('--validation', type=Path, required=True, help='separate local .npy validation array')
    parser.add_argument('--output', type=Path, required=True, help='empty output directory, preferably outside the repository')
    parser.add_argument('--steps', nargs='+', type=float, default=[80,120,180,270,400,600,900,1350,2000,3000,4500,6500])
    parser.add_argument('--bands', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.train.resolve() == args.validation.resolve():
        parser.error('training and validation must be distinct files')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    torch.set_num_threads(4)
    model, history = train_model(np.load(args.train, allow_pickle=False),
                                np.load(args.validation, allow_pickle=False), args.steps,
                                bands=args.bands, epochs=args.epochs, batch_size=args.batch_size,
                                device=args.device)
    save_model(model, args.output, metadata=history)
    print('Saved caller-trained model to', args.output)


if __name__ == '__main__':
    main()
