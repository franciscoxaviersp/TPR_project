import argparse

def getstream(line):
    return line.split(" ")[2].split("-")[0]


parser = argparse.ArgumentParser()
parser.add_argument("--input",required=True, help="input data file")
parser.add_argument("--output",required=True, help="output directory to store stream data files")
args = parser.parse_args()
inFile = args.input
dir = args.output

data = open(inFile, 'r')
lines = data.readlines()
lines = [line.rstrip() for line in lines]
print(lines[0])
first_stream = getstream(lines[0])

for line in lines:
    if "Stream" in line:
        stream_no = getstream(line)
        write = open(dir+'/'+stream_no+".txt",'w')
        continue
    write.write(line+"\n")


