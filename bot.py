import time
import random
import os
import keyboard

def write(word):
    for letter in word:
        time.sleep(random.randint(150,250)/1000)
        keyboard.press_and_release(letter)

p = 'password'
commands = ['dir', 'ipconfig', 'netstat /aon', 'powercfg /a', 'systeminfo','tasklist','driverquery','ping google.com'] # windows commands

for i in range(100):
    time.sleep(2)

    keyboard.write("ssh user@host")
    keyboard.press_and_release('enter')
    time.sleep(2)

    write(p)
    keyboard.press_and_release('enter')
    time.sleep(2)

    print("Opened session "+str(i))

    initTime = time.perf_counter() 
    currentTime = initTime

    sessionDuration = random.randint(15,240) #session will last between 15 and 180 seconds
    print("Next session duration: "+str(sessionDuration))

    while currentTime - initTime < sessionDuration:
        print("Current session duration:" +str(currentTime - initTime)+ " of "+str(sessionDuration)) 
        wait = random.randint(3,40)
        print("Waiting "+str(wait)+" seconds to send next command")
        time.sleep(wait) # duration between commands is between 2 and 40 seconds

        command = commands[random.randint(0,len(commands)-1)]

        print("Command: "+command)
        write(command)
        keyboard.press_and_release('enter')
        currentTime = time.perf_counter() 

    time.sleep(random.randint(3,40))
    write('exit')
    keyboard.press_and_release('enter')
    print ('Session '+str(i)+ ' closed')
    
    time.sleep(5)



