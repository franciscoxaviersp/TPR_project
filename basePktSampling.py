import sys
import argparse
import datetime
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark

def pktHandler(timestamp,srcIP,dstIP,lengthIP,file,sampDelta=1):
    global npkts
    global T0
    global outc
    global last_ks
    global source
    global dest
    
    
    print(timestamp,srcIP,dstIP,lengthIP)
    #if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):
    if npkts==0:
        T0=float(timestamp)
        last_ks=0
    print(T0,timestamp)            
    ks=int((float(timestamp)-T0)/sampDelta)
    if ks>last_ks:
        print('{} {} {} {} {}'.format(last_ks,*outc))
        file.write(' '.join([str(elem) for elem in outc]))
        file.write('\n')
        print("written")
        outc=[0,0,0,0]  
            
    if ks>last_ks+1:
        for j in range(last_ks+1,ks):
            print('{} {} {} {} {}'.format(j,*outc))
            file.write(' '.join([str(elem) for elem in outc]))
            file.write('\n')
            print("written")
                    
        
    if srcIP == source:
        outc[0]=outc[0]+1
        outc[1]=outc[1]+int(lengthIP)

    if srcIP == dest:
        outc[2]=outc[2]+1
        outc[3]=outc[3]+int(lengthIP)
    
    last_ks=ks
    npkts=npkts+1

def get_streams(pcap):

    streams = []
    file = open("streams.txt",'w') 
    for pkt in pcap:
        stream = int(pkt['tcp'].stream)
        if stream not in streams:
            streams.append(stream)
            file.write(str(stream)+'\n')

    print("Number of tcp streams: "+str(len(streams)))
    file.close()
    pcap.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-s', '--set_streams', nargs='?',required=False, help='set streams to file')
    parser.add_argument('-g', '--get_streams', nargs='?',required=True, help='get streams from file')
    
    args=parser.parse_args()

    global source
    global dest

    fileInput=args.input
        
    global npkts
    global T0
    global outc
    global last_ks

    streams_file = args.set_streams

    if streams_file:
        pcap = pyshark.FileCapture(fileInput, display_filter='ssh && !tcp.analysis.spurious_retransmission && !tcp.analysis.retransmission && !tcp.analysis.fast_retransmission')
        
        print('... Getting streams from pcap:')
        get_streams(pcap)
        

    streams_file = args.get_streams
    print("... Reading streams from")
    streams = open(streams_file,'r')
    file = open("sshdata2.txt",'w') 
    for stream in streams:
        npkts=0
        outc=[0,0,0,0]
        sampDelta=1
        #string = 'ssh && !tcp.analysis.spurious_retransmission && !tcp.analysis.retran
        print("Stream no: "+ str(stream).rstrip() +"--------------------------------------\n")
        file.write("Stream no: "+ str(stream).rstrip() +"--------------------------------------\n")
        
        for pkt in capture:
            timestamp,srcIP,dstIP,lengthIP=pkt.sniff_timestamp,pkt['ip'].src,pkt['ip'].dst,pkt['ip'].len
            print(timestamp)
            if i:
                source = srcIP
                dest = dstIP
                i = False

            pktHandler(timestamp,srcIP,dstIP,lengthIP,file,sampDelta)

    file.close()
    streams.close()
    
if __name__ == '__main__':
    main()
