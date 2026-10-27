"""Build the optional source-only PocketFFT adapter with GCC or Clang."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'), help='GCC/Clang C++ compiler executable')
    parser.add_argument('--output', type=Path, help='override library output path')
    args = parser.parse_args()
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise SystemExit('Compiler not found. Install GCC/Clang or pass --cxx /path/to/compiler.')
    root = Path(__file__).resolve().parents[1]
    source = root/'scec/native/cached_dct.cpp'
    suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    output = args.output or source.with_name('cached_dct'+suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ['-O3', '-std=c++11', '-fno-fast-math', '-ffp-contract=off']
    if sys.platform == 'win32':
        flags += ['-shared', '-static', '-static-libgcc', '-static-libstdc++']
    elif sys.platform == 'darwin':
        flags += ['-dynamiclib', '-fPIC']
    else:
        flags += ['-shared', '-fPIC']
    env = os.environ.copy()
    env['PATH'] = str(Path(compiler).parent)+os.pathsep+env.get('PATH', '')
    subprocess.run([compiler, *flags, str(source), '-o', str(output)], check=True, env=env)
    print('Built', output)


if __name__ == '__main__':
    main()
