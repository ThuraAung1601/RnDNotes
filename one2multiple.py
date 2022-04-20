# split txt files
# Author : Thura Aung

import argparse
import glob

parser = argparse.ArgumentParser(description='Concatenate Text Files')
parser.add_argument('-i', '--input', type=str, help='Input file', required=True)
parser.add_argument('-o', '--output', type=str, help='Output files path', required=True)

args = parser.parse_args()
inFile = getattr(args, 'input')
outPath = getattr(args, 'output')

fn = open(inFile,"r")
for i, line in enumerate(fn):
    f = open("{}/line_{}.txt".format(outPath,i),'w')
    f.write(line)
    f.close()
