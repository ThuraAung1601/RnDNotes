# concatenate txt files
# Author : Thura Aung

import argparse
import glob

parser = argparse.ArgumentParser(description='Concatenate Text Files')
parser.add_argument('-p', '--path', type=str, help='Path to text files', required=True)
parser.add_argument('-o', '--output', type=str, help='Concatenated text file', required=True)

args = parser.parse_args()
path = getattr(args, 'path')
out = getattr(args, 'output')

read_files = glob.glob("{}/*.txt".format(path))

with open(out, "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
