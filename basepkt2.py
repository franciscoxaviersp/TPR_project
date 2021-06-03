import sys
import datetime
import pyshark


def main():
    pcap = 'asdasd.pcapng'
    '''
    pcap = pyshark.FileCapture(pcap, display_filter='ssh')
    
    streams = []
    file = open("streamsteste2.txt",'w')
    for pkt in pcap:
        stream = int(pkt['tcp'].stream)
        if stream not in streams:
            streams.append(stream)
            file.write(str(stream)+'\n')

    print("Number of tcp streams: "+str(len(streams)))
    file.close()
    pcap.close()
    return
    '''

    streams = open('streamsteste2.txt','r').readlines()
    file = open("sshdatateste2.txt",'w')
    
    sampDelta=1
    streams = [stream.rstrip() for stream in streams] 
    for stream in streams:
        count = [0,0,0,0]
        npkts = 0
        print("Stream no: " + stream +"--------------------------------------\n")
        file.write("Stream no: " + stream +"--------------------------------------\n")
        capture = pyshark.FileCapture(pcap, display_filter='tcp.stream == {}'.format(stream))
        source = capture[0]['ip'].src
        dest = capture[0]['ip'].dst
        T0 = float(capture[0].sniff_timestamp)
        last_ks = 0

        for pkt in capture:
            timestamp, srcIP, dstIP, lengthIP = pkt.sniff_timestamp, pkt['ip'].src, pkt['ip'].dst, pkt['ip'].len
            ks=int((float(timestamp)-T0)/sampDelta)
            if ks>last_ks:
                print('{} {} {} {} {}'.format(last_ks,*count))
                file.write(' '.join([str(elem) for elem in count]))
                file.write('\n')
                count=[0,0,0,0]  
                    
            if ks>last_ks+1:
                for j in range(last_ks+1,ks):
                    print('{} {} {} {} {}'.format(j,*count))
                    file.write(' '.join([str(elem) for elem in count]))
                    file.write('\n')
            
            if srcIP == source:
                count[0]=count[0]+1
                count[1]=count[1]+int(lengthIP)

            if srcIP == dest:
                count[2]=count[2]+1
                count[3]=count[3]+int(lengthIP)
            
            last_ks=ks
            npkts=npkts+1
        capture.close()
    streams.close()
    file.close
          
            

    
if __name__ == '__main__':
    main()
