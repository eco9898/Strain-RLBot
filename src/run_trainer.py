import os, signal
from time import sleep, time
import subprocess
import re
from trainer_classes import getRLInstances, minimiseRL

pythonDir = os.getenv('LOCALAPPDATA') + '\RLBotGUIX\Python37\python.exe'
fileDir = 'src/trainer.py'
num_instances = 5

#Needs to be changed to a seperate thread and a pipe so it can be killed after an amount of time
def readLinesWait(wait_secs: int = -1, break_line: str = "NxUxLxL", break_strings = ["NxUxLxL"], lines_to_read: int = -1, print_output = True, ignore_trainer = True):
    lines = []
    if wait_secs > 0 or lines_to_read > 0:
        start = time()
        line = ""
        while (wait_secs != -1 and time() - start < wait_secs) or lines_to_read != 0:# or line != "":
            line = str(p.stdout.readline())[2:-5]
            if line != "" and line[0] == ">" and print_output and ignore_trainer:
                print(">" + line)
                #ignore trainer output
            elif line != "":
                lines.append(line)
                if print_output:
                    if line[0] == ">":
                        print(">" + line)
                    else:
                        print(line)
            if line == break_line:
                break
            foundStr = False
            for string in break_strings:
                if string in line:
                    foundStr = True
                    break
            if foundStr:
                break
            sleep(0.1)
            if lines_to_read > 0:
                lines_to_read -=1
    return lines

while True:
    print(">Starting trainer")
    subprocess.PIPE
    p = subprocess.Popen(pythonDir + " " + fileDir + " " + str(num_instances), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    wait_time = 0
    lines = []
    while len(lines) == 0:
        try:
            lines = readLinesWait(lines_to_read=1, print_output=False, ignore_trainer=False) # wait for PIPE to open
        except:
            lines = []
    try:
        wait_time = int(lines[0].replace(">Wait time: ", ""))
        lines.pop(0)
    except:
        wait_time = 22
    print(">Setup-info:")
    print(">Wait time:         ", wait_time)
    count = 0
    offset = 0
    #Wait until setup is printed
    readLinesWait(10, break_strings=["MMR"])
    while count < num_instances:
        start = time()
        print(">Parsing instance:" , (count + 1))
        lines.extend(readLinesWait(10, break_line="Done"))
        curr_count = 0
        while len(lines) > 0:
            m = re.search('Found (.+?) processes', lines.pop(0))
            if m:
                curr_count = int(m.group(1))
                break
        if count == 0 and curr_count > 0:
            offset = curr_count - 1
        if curr_count > count:
            count = curr_count - offset
            count = curr_count
            print(">Instances found:" , count)
            if count == num_instances:
                break
            lines.extend(readLinesWait(wait_time - (time() - start), break_strings=["Launching Rocket League"]))#, "CUDA", "Loaded previous"]))
        else:
            break
    done = False
    if count == num_instances:
        print(">Waiting to start")
        #this will block and is pointless unless an error is actually thrown, if trainer just hangs this wont stop restart it until it crashes
        lines.extend(readLinesWait(wait_time*2, break_strings=[">Training for"], ignore_trainer=False))
        while len(lines) > 0:   
            m = re.search('>Training for (.+?) timesteps', lines.pop(0))
            if m:
                done = True
    if count != num_instances or not done:
        print(">Killing trainer")
        p.kill()
        PIDs = getRLInstances()
        while len(PIDs) > 0:
            pid = PIDs.pop()["pid"]
            print(">Killing RL instance", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                print(">Failed")
    else:
        minimiseRL()
        try:
            print(">Finished parsing trainer")
            while True:
                readLinesWait(1)
        except KeyboardInterrupt:
            try:
                os.kill(p.pid, signal.CTRL_C_EVENT)
                sleep(1) #wait to receive kill signal as well
                break
            except KeyboardInterrupt:
                readLinesWait(10, "Save complete")
                break
