import pygame
from pygame.locals import * 
import pyaudio, wave
import pylink
import time
import traceback
import threading
import sys
from tests.testFunctions import testTiming,compareEyeTrackingWithBeh,prepareEyeTrackingTiming 
from classes.fileIDInput import createOutputs

# Runtime diagnostics: thread dumps and global exception hooks
def dump_all_thread_traces(log_path="runtime_trace.txt"):
    try:
        frames = sys._current_frames()
        with open(log_path, 'a', encoding='utf8') as f:
            f.write(f"\n\n=== THREAD DUMP {time.asctime()} ===\n")
            for tid, frame in frames.items():
                f.write(f"\n--- Thread id: {tid} ---\n")
                traceback.print_stack(frame, file=f)
    except Exception:
        pass

def exception_hook(exc_type, exc_value, exc_tb):
    try:
        tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            outputControll.write(f"UNCAUGHT EXCEPTION:\n{tb}")
        except Exception:
            pass
        dump_all_thread_traces()
        with open("runtime_exception.txt","a",encoding='utf8') as fh:
            fh.write(f"=== EXCEPTION {time.asctime()} ===\n")
            fh.write(tb)
            fh.write("\n")
    except Exception:
        pass
    # call default handler too
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = exception_hook

def _threading_excepthook(args):
    try:
        exception_hook(args.exc_type, args.exc_value, args.exc_traceback)
    except Exception:
        pass

# Python 3.8+ exposes threading.excepthook
try:
    threading.excepthook = _threading_excepthook
except Exception:
    pass
from classes.CalibrationGraphivs import CalibrationGraphics
from classes.Audio import audioTrial
from classes.Audio import storyTimeDict3,storyTimeDict3a,storyTimeDict3b,storyTimeDictTest
from classes.recallTrial import recallTrial
from classes.welcomeMessage import welcomeMessage
from classes.welcomeMessage import generateMessages

from  random import choice
import pandas as pd
import pickle
import sys,os,subprocess
import datetime

 
#### ----- Setups ------ ####

#Getting User Info:
outputControll = createOutputs(dataPath=".\\Experiment\\data");
outputControll.idInputwin()
outputControll.write("\n       ###### STRATEGIC MW EXPERIMENT v0.98 ######    \n")
outputControll.write(f"  Log Time: {datetime.datetime.now()}")
outputControll.write(f"  Currently Processing Subject N.: {outputControll.ID}")
if outputControll.ID == "" or len(outputControll.ID) < 3:
     raise ValueError("Invalid ID - Exiting PRocedure\n\n") # To Do XD 

# Display Information
pygame.init()
disp = pylink.getDisplayInformation()
print(disp)
outputControll.write("  Current Display Information: ")
outputControll.write(f"      Width:  {disp.width}")
outputControll.write(f"      Height: {disp.height}\n")

width = disp.width
height =disp.height
screen = pygame.display.set_mode((width, height),
                                 pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
tempInitialTime = 0; # For Gathering Realtive Timestamps
# PyGame setups
font = pygame.font.Font(None, 50)
pygame.mouse.set_visible(False)
pygame.display.set_caption("Strategic MW Experiment")

# story Global parameters Setup:
storyPart = "welcome1"  # Controlling the experiment flow
dummyMode = True
TestMode = False

SCREEN_WIDTH_CM = 53 #Width
SCREEN_HEIGHT_CM = 30 # Height 
VIEWING_DISTANCE_CM = 93 # 
entityName = [storyTimeDict3["partNames"][0][:-2], storyTimeDict3["partNames"][1][:-2]][int(choice([0, 1]))]
outputControll.write(f"      Viewing Distance:  {VIEWING_DISTANCE_CM}\n")

outputControll.write("  Experiment Config: ")
outputControll.write(f"      dummyMode:             {dummyMode}")
outputControll.write(f"      Current Named Entity:  {entityName}")
outputControll.write(f"      Test Mode:             {TestMode}")

### Make Experiment Objects

# Experiment  Message Initialization
[WelcomeMessage1, WelcomeMessage11, WelcomeMessage2,WelcomeMessage2b,WelcomeMessage2c, WelcomeMessage3,WelcomeMessage21alt,WelcomeMessage4,exitMessage1] = generateMessages(entityName)
                                            # Message dictionary                               Font, Screen, Next exp Part, prev exp Part
welcome = welcomeMessage([WelcomeMessage1,WelcomeMessage11, WelcomeMessage2,WelcomeMessage2b,  WelcomeMessage2c],font,screen,"welcome1","practicerun")
welcome1 = welcomeMessage([WelcomeMessage3],font,screen,"welcome_cal","calibration1")

welcome2 = welcomeMessage([WelcomeMessage21alt],font,screen,"welcome2","story1")
calib2 = welcomeMessage([WelcomeMessage4],font,screen,"calib_text","calibration2")
exitMessage = welcomeMessage([exitMessage1],font,screen,"exit","")

# Audio File initialization:
if TestMode:
    Story_practice = audioTrial(r".\TextToSpeech\StoryTest_AI",storyTimeDictTest,font,screen,"practicerun","welcome_cal",outputControll,verbose=2,practice=True) # PracticeRun

    Story1 = audioTrial(r".\TextToSpeech\Story3_AIpartTest1",storyTimeDict3a,font,screen,"story1","calib_text",outputControll,verbose=2)
    Story2 = audioTrial(r".\TextToSpeech\Story3_AIpartTest2",storyTimeDict3b,font,screen,"story2","recall1",outputControll,verbose=2)
else:
    Story_practice = audioTrial(r".\TextToSpeech\StoryTest_AI",storyTimeDictTest,font,screen,"practicerun","welcome_cal",outputControll,verbose=2,practice=True) #PracticeRun


    Story1 = audioTrial(r".\TextToSpeech\Story3_AIpart1",storyTimeDict3a,font,screen,"story1","calib_text",outputControll,verbose=2)
    Story2 = audioTrial(r".\TextToSpeech\Story3_AIpart2",storyTimeDict3b,font,screen,"story2","recall1",outputControll,verbose=2)
# Audio Recording Object Initialization:
recall1 = recallTrial("story1.wav",font,screen,"recall1","exit",outputControll,entityName)

# connect to the tracker
if not dummyMode:
    el_tracker = pylink.EyeLink('100.1.1.1')

    # open an EDF data file on the Host PC
    outputControll.manageEyeTrackingFile(el_tracker)

    # send over a command to let the tracker know the correct screen resolution
    scn_coords = "screen_pixel_coords = 0 0 %d %d" % (width - 1, height - 1)
    el_tracker.sendCommand(scn_coords)
    el_tracker.sendCommand(f"screen_phys_coords = -{SCREEN_WIDTH_CM/2*10} {SCREEN_HEIGHT_CM/2*10} {SCREEN_WIDTH_CM/2*10} -{SCREEN_HEIGHT_CM/2*10}")

    # Send viewing distance (in millimeters)
    el_tracker.sendCommand(f"screen_distance = {VIEWING_DISTANCE_CM*10}")

    # Instantiate a graphics environment (genv) for calibration
    genv = CalibrationGraphics(el_tracker, screen)
else:
    el_tracker = pylink.EyeLink(None)

outputControll.write("\nEXPERIMENTS STARTS")



### ------ Experiment ------- ####
el_tracker.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")
el_tracker.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON")
el_tracker.sendCommand("pupil_size_diameter = YES")
prev_ts = time.perf_counter()
running = True
begBlockFlag = True
while running: 
    # Watchdog for main-loop stalls
    now = time.perf_counter()
    if now - prev_ts > 0.2:
        outputControll.write(f"Main loop stall: {now - prev_ts:.3f}s")
    prev_ts = now

    # Pre-process some events (allow F12 manual dump, ignore double-clicks, detect window changes)
    events = pygame.event.get()
    for ev in events:
        handled = False
        # Manual thread dump (press F12 while reproducing the issue)
        if ev.type == pygame.KEYDOWN and getattr(ev, 'key', None) == pygame.K_F12:
            dump_all_thread_traces()
            outputControll.write("Manual thread dump (F12) written")
            handled = True

        # Ignore double-clicks (some platforms expose .clicks)
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            if getattr(ev, 'clicks', 1) > 1:
                outputControll.write("Ignored double-click")
                handled = True

        # Detect window resize / mode changes and re-apply fullscreen
        if ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            outputControll.write(f"Window event detected: {ev}")
            screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.mouse.set_visible(False)
            handled = True

        # Re-post unhandled events so existing code can process them
        if not handled:
            try:
                pygame.event.post(pygame.event.Event(ev.type, ev.dict))
            except Exception:
                pass

    screen.fill((127, 127, 127))  # Clear screen before each frame
    ### Initial Instructions ###
    if storyPart == "welcome1":
       if begBlockFlag:
            begBlockFlag = False

       storyPart = welcome.run()

       if storyPart != "welcome1":
            begBlockFlag = True

    ### practice Run of Experiments (3 trials) ###
    elif storyPart == "practicerun":

        if begBlockFlag:
            outputControll.write(f"\nPractice Run Beginning ({time.time():.3f})\n")

            begBlockFlag = False

        storyPart =  Story_practice.run()
        if storyPart != "practicerun":
            outputControll.write(f"\nPractice Run Ending ({time.time():.3f})\n")

            begBlockFlag = True
    ### First Calibration Welcome and Execution:
    elif storyPart == "welcome_cal":

        if begBlockFlag:

            begBlockFlag = False

        storyPart = welcome1.run()
        if storyPart != "welcome_cal":  
            begBlockFlag = True

    elif storyPart == "calibration1":

        tempInitialTime = time.time();
        outputControll.write(f"\nCalibration of EyeTracker ({time.time():.3f})\n")

        if not dummyMode:
            try:
                pylink.closeGraphics()

                pylink.pumpDelay(50)
                pylink.openGraphicsEx(genv)  # Register CalibrationGraphics

                el_tracker.imageModeDisplay()

                el_tracker.sendCommand("automatic_calibration_pacing = 1000")  # Pacing of Targets - allow for automaticity
       
                el_tracker.doTrackerSetup()  # Calibration Setup 

                pylink.pumpDelay(50)            
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()     

        outputControll.write(f"   Calibration Ended (Duration: {time.time()-tempInitialTime})")

        storyPart = "welcome2"
        begBlockFlag = True

        screen.fill((255/2, 255/2, 255/2))  # Reset screen
        pygame.display.flip()  # Ensure Pygame updates after calibration

    ### Last message before Beginning      
    elif storyPart == "welcome2":
       
        if begBlockFlag:
            begBlockFlag = False

        storyPart = welcome2.run()

        if storyPart != "welcome2":
           begBlockFlag = True
        

    ### First Part of Story Trials (20 trials before Calibration!)
    elif storyPart == "story1":
        if begBlockFlag:
            el_tracker.startRecording(1, 1, 1, 1)
            pylink.pumpDelay(100)  # Small delay to ensure recording starts

            outputControll.write(f"\nStarting Audio Trial 1 ({time.time():.3f}) *** ")
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = Story1.run()
        if storyPart != "story1":
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tEND")
            outputControll.write("\nAnswer Timing Log:")
            outputControll.write(pd.DataFrame(Story1.timingLog,columns=["Event","StoryTime","RT","partDuration"]).to_string())
            begBlockFlag = True
            testTiming(Story1.timingLog,outputControll)

    ### Second Calibration ()
    elif storyPart == "calib_text":
        pygame.mouse.set_visible(False)

        if begBlockFlag:
           begBlockFlag = False

        storyPart = calib2.run()

        if storyPart != "calibration2":
           begBlockFlag = True
        

    elif storyPart == "calibration2":
        tempInitialTime = time.time();
        outputControll.write(f"\nCalibration of EyeTracker ({time.time():.3f})\n")

        if not dummyMode:
            try:
              
                el_tracker.sendCommand("automatic_calibration_pacing = 1000")  # Pacing of Targets - allow for automaticity
                el_tracker.doTrackerSetup()  # Calibration Setup 

                pylink.pumpDelay(50)            
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()     

        outputControll.write(f"   Calibration Ended (Duration: {time.time()-tempInitialTime})")

        storyPart = "story2"
        begBlockFlag = True

        screen.fill((255/2, 255/2, 255/2))  # Reset screen
        pygame.display.flip()  # Ensure Pygame updates after calibration

    ### Second part of experiment (20 trials after calibration)
    elif storyPart == "story2":
        #insert my AudioClass.run() Here!!!
        if begBlockFlag:
            el_tracker.startRecording(1, 1, 1, 1)
            pylink.pumpDelay(100)  # Small delay to ensure recording starts

            outputControll.write(f"\nStarting Audio Trial 1 ({time.time():.3f}) *** ")
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = Story2.run()
        if storyPart != "story2":
            outputControll.writeToEyeLink("\tLISTEN\tSTORY_1\tEND")
            outputControll.write("\nAnswer Timing Log:")
            outputControll.write(pd.DataFrame(Story2.timingLog,columns=["Event","StoryTime","RT","partDuration"]).to_string())
            begBlockFlag = True
            testTiming(Story2.timingLog,outputControll)

    ### General Story Recall
    elif storyPart == "recall1":
        pygame.mouse.set_visible(False)

        if begBlockFlag:
            el_tracker.stopRecording()

            outputControll.write(f"\nRecall of Audio  1 ({time.time():.3f} *** \n")

            outputControll.writeToEyeLink("\tRECALL\tSTORY_1\tBEG")
            begBlockFlag = False

        storyPart = recall1.run()

        if storyPart != "recall1":
            outputControll.writeToEyeLink("\tRECALL\tSTORY_1\tEND")
            begBlockFlag = True

    ### End of Experiment message and writing
    elif storyPart == "exit":
        if begBlockFlag:
            outputControll.write(f"\nEND OF A PROCEDURE ({time.time():.3f}) *** \n")

            outputControll.writeToEyeLink("\tPROCEDURE\t0\tEND")

            begBlockFlag = False

        storyPart = exitMessage.run()

        if storyPart != "exit":
           begBlockFlag = True  
    else:
        running = False

    pygame.display.flip()  # Update display

#### --- Saving and Tests --- ####
if not dummyMode:
    edf_path = os.path.abspath(os.path.join(outputControll.dataPath, outputControll.ID))
    if not os.path.exists(edf_path):
        os.makedirs(edf_path)  # 

    # Save Data to a pickle
    logs = {'story1_beh': pd.DataFrame(Story1.timingLog,columns=["Event","StoryTime","RT","partDuration"])}
    with open(os.path.join(edf_path,f'{outputControll.ID}.pickle'), 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filePath = os.path.join(edf_path,f'{outputControll.ID}.edf');
    el_tracker.closeDataFile()  # Close the EDF file
    el_tracker.receiveDataFile(f'{outputControll.ID}.edf',os.path.join(edf_path,f'{outputControll.ID}.edf'))
    outputControll.write(f"Eye Tracking data saved to: {filePath}")
    asciiFilePath = outputControll.edf2ascii(filePath);

    ### TESTS
    EyetrackerDF = prepareEyeTrackingTiming(os.path.join(edf_path,f'{outputControll.ID}.asc'),outputControll)
    compareEyeTrackingWithBeh(EyetrackerDF,logs['story1_beh'],logs['story2_beh'],outputControll)
    outputControll.write("Testing EDF file Timing")
    outputControll.write("   Story1: ")
    testTiming(EyetrackerDF['story1'].to_numpy(),outputControll)
    outputControll.write("   Story2: ")
    testTiming(EyetrackerDF['story2'].to_numpy(),outputControll)

    el_tracker.close()  # Disconnect from EyeLink

pygame.quit()
exit()

 