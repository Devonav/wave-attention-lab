import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description='Compare fixed and dimension-sized LDS.')
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
for trial in range(3):
    for causal in (False, True):
        for n in (128, 256, 512, 1024):
            for d in (64, 128):
                # Alternate ordering to reduce systematic warm-up/drift bias.
                order = ('baseline', 'candidate') if trial % 2 == 0 else ('candidate', 'baseline')
                for version in order:
                    print(f'trial={trial + 1} causal={causal} N={n} D={d} {version}', flush=True)
                    cmd = [str(binaries[version]), '--backend', 'hip', '--algorithm', 'tiled',
                           '--heads', '8', '--sequence', str(n), '--dimension', str(d),
                           '--warmup', '10', '--iterations', '100', '--verify',
                           '--csv', str(out / f'{version}.csv')]
                    if causal:
                        cmd.append('--causal')
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
for version, binary in binaries.items():
    for d in (64, 128):
        subprocess.run(['/opt/rocm/bin/rocprofv3', '--kernel-trace', '--output-format', 'csv',
                        '--output-directory', str(out / f'{version}-d{d}'), '--', str(binary),
                        '--backend', 'hip', '--algorithm', 'tiled', '--heads', '8',
                        '--sequence', '1024', '--dimension', str(d),
                        '--warmup', '2', '--iterations', '10'], check=True)
