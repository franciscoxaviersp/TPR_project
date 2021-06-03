def getstream(line):
    return line.split(" ")[2].split("-")[0]


data = open('sshdata5.txt', 'r')
lines = data.readlines()
lines = [line.rstrip() for line in lines]
first_stream = getstream(lines[0])

for line in lines:
    if "Stream" in line:
        stream_no = getstream(line)
        write = open('streams05/'+stream_no+".txt",'w')
        continue
    write.write(line+"\n")


