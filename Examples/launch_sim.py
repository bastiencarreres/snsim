import snsim
import argparse
import numba

parser = argparse.ArgumentParser()
parser.add_argument("config_file",
                    default=None,
                    type=str,
                    help="Configuration file")

parser.add_argument('-print_config', action='store_true',
                    help='Print the config file')


parser.add_argument('--nb_threads', default=None, type=int,
                    help='Number of thread numba can use')


args = parser.parse_args()

# Numba parallel options
if args.nb_threads is not None:
    numba.set_num_threads(args.nb_threads)

print(f'Numba is configured to use {numba.get_num_threads()} threads')

if __name__ == '__main__':
    sim = snsim.Simulator(args.config_file, print_config=args.print_config)
    sim.simulate()
