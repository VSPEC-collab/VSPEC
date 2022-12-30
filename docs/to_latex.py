from os import chdir
from pathlib import Path

WORKING_DIRECTORY = Path(__file__).parent

chdir(WORKING_DIRECTORY)

infilename = 'param_descriptions.csv'
outfilename = 'param_descriptions.txt'
with open(infilename,'r') as infile:
    with open(outfilename,'w') as outfile:
        header = infile.readline()
        outfile.write(header)
        for line in infile:
            name,data_type,desc = line.split(',')
            name = f'\\verb|{name}|'
            outfile.write(f'{name} & {data_type} & {desc.replace("\n","")} \\\\ \n')