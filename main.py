"""Reader entry point: construction, spectra, decoding, experiments and checks."""
import argparse
import json
from pathlib import Path
import sys
import time

from polar_subcodes.code import load_code
from polar_subcodes.codec import load_codec
from polar_subcodes.io import ROOT, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__,epilog='See README.md and docs/ for custom profiles and support conventions.')
    commands = p.add_subparsers(dest='command',required=True)
    commands.add_parser('examples',help='List all experiment and utility entry points')
    r = commands.add_parser('reproduce',help='Run a legacy construction; see reproduction/ for Section VI')
    from polar_subcodes.experiments.reproduce import CASES
    r.add_argument('case',choices=[*CASES,'all'])
    r.add_argument('--output',type=Path)
    r.add_argument('--seconds',type=float,default=600)
    r.add_argument('--nodes',type=int,default=1_000_000)
    r.add_argument('--work',type=int,default=100_000_000)
    r.add_argument('--repetitions',type=int,default=1,help='Sequential full design replays')
    c = commands.add_parser('construct',help='Try the deterministic first domain for your own profile')
    c.add_argument('--profile',type=Path,required=True)
    c.add_argument('--distance',type=int,required=True)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--recipe',choices=['auto','word','nested','gap'],default='auto')
    c.add_argument('--seconds',type=float,default=600)
    c.add_argument('--nodes',type=int,default=1_000_000)
    c.add_argument('--work',type=int,default=100_000_000)
    c.add_argument('--distance-only',action='store_true',help='Require exclusion and a witness, do not count the target coefficient')
    d = commands.add_parser('inspect',help='Print the information set and dynamic frozen equations')
    d.add_argument('input',type=Path)
    d.add_argument('--include-static',action='store_true')
    d.add_argument('--output',type=Path)
    e = commands.add_parser('enumerate',help='Exact decomposition counts or a complete bounded-weight word list')
    e.add_argument('--support',type=Path,required=True)
    e.add_argument('--cutoff',type=int,required=True)
    e.add_argument('--method',choices=['decomposition','words','native','python','quotient','direct'],default='decomposition')
    e.add_argument('--work',type=int,default=100_000_000)
    e.add_argument('--output',type=Path,required=True)
    s = commands.add_parser('simulate',help='BPSK-AWGN Monte Carlo with a general SCL decoder')
    s.add_argument('--support',type=Path,required=True)
    s.add_argument('--snr',type=float,default=3.0,help='Eb/N0 in dB')
    s.add_argument('--frames',type=int,default=1000)
    s.add_argument('--list-size',type=int,default=64)
    s.add_argument('--batch-size',type=int,default=32)
    s.add_argument('--seed',type=int,default=2026)
    s.add_argument('--backend',choices=['auto','numpy','cpp'],default='auto')
    s.add_argument('--messages',choices=['zero','random'],default='zero')
    s.add_argument('--output',type=Path,required=True)
    commands.add_parser('verify-theory',help='Run exhaustive small-code checks and implementation regressions')
    toy = commands.add_parser('toy',help='Reproduce exhaustive N32 support-domain experiments and figures')
    toy.add_argument('--output',type=Path,required=True)
    benchmark = commands.add_parser('benchmark-enumeration',help='Matched complete-word direct/quotient comparison')
    benchmark.add_argument('--output',type=Path,required=True)
    benchmark.add_argument('--repetitions',type=int,default=3)
    historical = commands.add_parser('intermediate',help='Verify the stored historical distance-20 illustration')
    historical.add_argument('--output',type=Path,required=True)
    plot = commands.add_parser('plot-run',help='Plot a completed outer conditional-mean trace')
    plot.add_argument('run',type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command=='examples':
        print('''Manuscript r112 (start here)
python -m paper_examples all
python -m theory.check_results
python -m reproduction list
python -m reproduction check
python -m reproduction construct --cases n64_k32 --output runs/fig5

Earlier five-case release and general utilities
python examples/n64_k32.py
python examples/n128_k42.py
python examples/n256_k64.py
python examples/n256_k128.py
python examples/n256_k171.py
python examples/toy_mechanisms.py --output runs/toy
python examples/enumeration_comparison.py --output runs/enumeration
python examples/repeated_designs.py --output runs/repeated --repetitions 3
python examples/intermediate_distance.py --output runs/intermediate
python main.py plot-run runs/n256
python main.py inspect results/n64_k32/support.json
python main.py construct --profile profile/n64_k32.json --distance 12 --output runs/custom
python main.py enumerate --support results/n64_k32/support.json --cutoff 12 --method words --output runs/words
python main.py enumerate --support results/n256_k128/support.json --cutoff 24 --method decomposition --output runs/count24
python main.py simulate --support results/n64_k32/support.json --snr 3 --frames 1000 --list-size 64 --output runs/fer
python main.py verify-theory''')
    elif args.command in ('construct','reproduce'):
        from polar_subcodes.construction.pipeline import run,Limits
        limits = Limits(args.seconds,args.nodes,args.work)
        if args.command=='construct':
            run(args.profile,args.distance,args.output,limits,args.recipe,not args.distance_only)
        else:
            from polar_subcodes.experiments.reproduce import reproduce,CASES
            if args.repetitions<1:
                raise ValueError('Repetitions must be positive')
            if args.repetitions>1:
                from polar_subcodes.experiments.replays import run as repeat
                repeat(args.output or ROOT/'runs'/f'repeated_{time.time_ns()}',
                       list(CASES) if args.case=='all' else [args.case],args.repetitions,limits)
            elif args.case=='all':
                base = args.output or ROOT/'runs'/f'all_{time.time_ns()}'
                for name in CASES:
                    reproduce(name,base/name,limits)
            else:
                reproduce(args.case,args.output,limits)
    elif args.command=='inspect':
        codec = load_codec(args.input)
        description = codec.mother.describe(args.include_static)
        if codec.polynomial is not None:
            description += f'\nCRC polynomial {codec.polynomial:#x}, payload {codec.payload} bits.'
        if codec.taps:
            description += f'\nPAC taps in increasing delay order: {list(codec.taps)}.'
        print(description)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(description+'\n',encoding='utf-8')
    elif args.command=='enumerate':
        from polar_subcodes.spectrum.api import enumerate_code
        enumerate_code(load_codec(args.support).as_subcode(),args.cutoff,args.output,args.method,args.work)
    elif args.command=='simulate':
        from polar_subcodes.decoding.simulation import simulate
        simulate(load_codec(args.support),args.output,args.snr,args.frames,args.list_size,
                 args.batch_size,args.seed,args.backend,args.messages)
    elif args.command=='verify-theory':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            return 1
    elif args.command=='toy':
        from polar_subcodes.experiments.toy_runner import run
        run(args.output)
    elif args.command=='benchmark-enumeration':
        from polar_subcodes.experiments.enumeration_benchmark import run
        run(args.output,args.repetitions)
    elif args.command=='intermediate':
        from polar_subcodes.spectrum.api import enumerate_code
        code=load_code(ROOT/'results/intermediate_n256_k128/support.json')
        record=enumerate_code(code,20,args.output,'native')
        if record['spectrum']!={0:1,20:4042}:
            raise AssertionError('Historical distance-20 spectrum mismatch')
    elif args.command=='plot-run':
        from polar_subcodes.experiments.trace_plot import plot
        plot(args.run)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        sys.exit(2)
