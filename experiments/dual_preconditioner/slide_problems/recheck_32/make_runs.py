"""Four slides with an identical setup (bubble array, compression F11: 1 -> 0.6, regular grid): one copy of the
contact script per slide, differing only in the lines below.

usage: python make_runs.py N [--adapted] [--gc GC] [--preconditioner Green] [--device gpu] [--regularization medium]
       (folders N<N>[_adapted][_Green][_gpu][_reg-medium]/)"""
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, nargs='?', default=32)
parser.add_argument('--adapted', action='store_true', help='grid adapted to the bubbles (default: regular grid)')
parser.add_argument('--gc', type=float, default=0.1, help='fracture toughness of slide 4')
parser.add_argument('--preconditioner', choices=('Green_Jacobi', 'Green', 'dual'), default='Green_Jacobi',
                    help='preconditioner of the mechanics CG')
parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
parser.add_argument('--gc-random', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                    help='slide 4: correlated random Gc field between LOW and HIGH (correlation length 0.1, seed 0)')
parser.add_argument('--regularization', choices=('everywhere', 'medium'), default='everywhere',
                    help='where the HuHu-LuLu term acts: everywhere, or in the third medium only')
args = parser.parse_args()
n = args.n
here = os.path.dirname(os.path.realpath(__file__))
script = os.path.join(here, 'example_2D_phase_field_fracture_contact_adapted_grid.py')   # the local copy
common = {
    r"^nnn = \d+": f"nnn = {n}",
    r"^device = '\w+'": f"device = '{args.device}'",
    r"^mechanics_preconditioner = '\w+'": f"mechanics_preconditioner = '{args.preconditioner}'",
    r"^regularization_region = '\w+'": f"regularization_region = '{args.regularization}'",
    r"^geometry_name = '\w+'": "geometry_name = 'bubble_array'",
    r"^H_macro = np\.array\(\[\[0\.0, 0\.0\],\n\s+\[0\.01, -0\.27\]\]\)": "H_macro = np.array([[0.0, 0.0],\n                    [0.0, -0.4]])",
    r"^nb_increments = \d+": "nb_increments = 100",
    r"^adapt_grid_to_geometry = \w+": f"adapt_grid_to_geometry = {args.adapted}",
    r"^make_figures = \w+": "make_figures = False",
    r"^output_root = .*": f"output_root = {os.path.join(here, 'runs')!r}",
    r"^sys\.path\.insert\(0, os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), '\.\./\.\.'\)\)\)":
        "sys.path.insert(0, '/home/martin/PycharmProjects/muFFTTO')",
    r"^grid_folder = .*": "grid_folder = " + repr(os.path.join(here, 'grids')),
}
slides = {
    '1_newton_cg': {r"^mechanics_solver = '\w+'": "mechanics_solver = 'newton_cg'",
                    r"^void_model = '\w+'": "void_model = 'linear'",
                    r"^solve_damage_problem = \w+": "solve_damage_problem = False"},
    '2_trust_region': {r"^void_model = '\w+'": "void_model = 'linear'",
                       r"^solve_damage_problem = \w+": "solve_damage_problem = False"},
    '3_contact': {r"^solve_damage_problem = \w+": "solve_damage_problem = False"},
    '4_fracture': {r"^Gc = [0-9.e+-]+ ": f"Gc = {args.gc:g} ",
                   **({r"^Gc_field = '\w+'": "Gc_field = 'random'",
                       r"^Gc_range = \(.*\)": f"Gc_range = ({args.gc_random[0]:g}, {args.gc_random[1]:g})"}
                      if args.gc_random else {})},
}
source = open(script).read()
for slide, changes in slides.items():
    text = source
    for pattern, replacement in {**common, **changes}.items():
        text, count = re.subn(pattern, replacement, text, flags=re.M)
        assert count >= 1, (slide, pattern)
    folder = os.path.join(here, f'N{n}' + ('_adapted' if args.adapted else '')
                          + {'Green': '_Green', 'dual': '_dual'}.get(args.preconditioner, '')
                          + ('_gpu' if args.device == 'gpu' else '')
                          + ('_reg-medium' if args.regularization == 'medium' else '')
                          + ('_Gc-random' if args.gc_random else ''), slide)
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, 'run.py'), 'w').write(text)
    print(folder)
