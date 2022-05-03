import os, signal
from time import sleep, time
import subprocess
import re
from trainer_classes import getRLInstances, minimiseRL
wait_time=22
num_instances = 10

pythonDir = os.getenv('LOCALAPPDATA') + '\RLBotGUIX\Python37\python.exe'
fileDir = 'src/trainer.py'


def readLinesWait(wait_secs: int, break_line: str = "Done", break_string: str = "training for "):
    lines = []
    if wait_secs > 0:
        start = time()
        line = ""
        while time() - start < wait_secs:# or line != "":
            line = str(p.stdout.readline())[2:-5]
            if line != "":
                lines.append(line)
                print(line)
            if line == break_line:
                break
            if break_string in line:
                break
            sleep(0.1)
    return lines

while True:
    print(">Starting trainer")
    subprocess.PIPE
    p = subprocess.Popen(pythonDir + " " + fileDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    count = 0
    offset = 0
    lines = []
    while count < num_instances:
        start = time()
        print(">Parsing instance:" , (count + 1))
        lines.extend(readLinesWait(10))
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
            lines.extend(readLinesWait(wait_time - (time() - start)))
        else:
            break
    done = False
    if count == num_instances:
        print(">Waiting to start")
        #this will block and is pointless unless an error is actually thrown, if trainer just hangs this wont stop restart it until it crashes
        lines.extend(readLinesWait(wait_time*2))
        while len(lines) > 0:
            m = re.search('training for (.+?) timesteps', lines.pop(0))
            if m:
                done = True
    if count != num_instances or not done:
        print(">Killing trainer")
        p.kill()
        PIDs = getRLInstances()
        while len(PIDs) > 0:
            pid = PIDs.pop()["pid"]
            print(">Killing RL instance", pid)
            os.kill(pid, signal.SIGTERM)
    else:
        minimiseRL()
        while True:
            print(">Finished parsing trainer")
            readLinesWait(1)
