import argparse


def get_single_argument(name: str= "-i"):
    parser = argparse.ArgumentParser(f"Input: {name}")
    parser.add_argument(name, nargs=1, type=int, dest="index")
    args = parser.parse_args()
    if args.index is not None:
        index = args.index[0]
    else:
        index = 0
    return index
