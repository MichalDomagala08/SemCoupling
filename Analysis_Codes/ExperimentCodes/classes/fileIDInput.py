import pygame
import sys
import os
import subprocess

import tkinter as tk
from tkinter import simpledialog


class createOutputs:
    """
        A Class that handles Creating Output Files bothb Behavioral as well as EyeTracking. 
        Also Handles Verbosity and Recordings to a proper Path
    """

    def __init__(self,dataPath,charLim=8,ID="",dummyMode=False,verbose=1):
        self.dataPath = dataPath; # Path to a Data File 
        self.charLim=charLim
        self.ID=ID

        #self.logfile = open(os.path.join(dataPath,ID,ID+"_log.txt"),'w+')
        self.terminal = sys.stdout  # Keep original stdout
        self.dummyMode = dummyMode  # Logs the dummy Mode from File 
        self.verbose = verbose;

    def idInputwin(self):
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Show an input dialog (this looks like a real Windows prompt!)
        user_input = simpledialog.askstring("Enter ID", "Please enter the participant ID:", parent=root)
        
        # Validate character limit
        if user_input and len(user_input) > self.charLim:
            user_input = user_input[:self.charLim]  # Trim to limit

        self.ID =user_input # Returns the ID entered or None if canceled
        if os.path.isdir(os.path.join(self.dataPath,user_input)):
            Warning("A Person With that exact Same ID has already been Processed!")
        else:
            os.mkdir(os.path.join(self.dataPath,user_input)) # Make a File when all will be Kept

        self.logfile =  open(os.path.join(self.dataPath,user_input,user_input+"_log.txt"),'w+')


    def write(self, message):
        self.terminal.write(message+"\n")  # Print to console
        self.terminal.flush()  # 👈 Ensure immediate output to the terminal

        self.logfile.write(message+"\n")  # Write to file
        self.logfile.flush()  # Ensure it gets written immediately

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


    def getTracker(self,el_tracker):
        """
            Saves for Precise Controll an el_tracker Object
        """
        self.el_tracker = el_tracker

    def writeToEyeLink(self, message):
        """
        Logs the start of a story part, its duration, and EyeLink timing.
        
        """
        try:
            if not self.dummyMode:
                timestamp = self.el_tracker.getCurrentTime()  # Get the current EyeLink time
                message_all = f"{message}"

                if timestamp - (23**2 -1) == 0: # Checks whether EyeTracker is correctly Outputed
                    raise ValueError("Eye Tracker Gives Erratic Output: Check Connection ot Dummy Mode\n\n") # 

                
                self.el_tracker.sendMessage(message_all)  # Send message to EyeLink
                if self.verbose == 2:
                    self.write(f"      [EyeLink] Logged: {message_all}")  # Debugging confirmation

        except Exception as e:
            self.write(f"      [EyeLink ERROR]: {e}")

    def manageEyeTrackingFile(self,el_tracker):
        el_tracker.openDataFile(f'{self.ID}.edf')
        self.write("Link established")
        self.el_tracker = el_tracker

    def edf2ascii(self,filepath,outputFolder = None):

        if outputFolder == None:
            outputFolder,_ = os.path.split(filepath) #the same output folder as in filepath

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        ascii_file = os.path.join(outputFolder, os.path.basename(filepath).replace(".edf", ".asc"))
        # Run EyeLink's edf2asc command
        command = f"edf2asc -y {filepath} -o {ascii_file}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if conversion was successful
        if result.returncode == 0:
            self.write(f"EDF converted successfully: {ascii_file}")
            return ascii_file
        else:
            self.write(f"Error converting EDF: {result.stderr}")
            return None


    def manageBEhavioralData(self,behData):
        filename = os.path.join(self.dataPath,self.ID,self.ID+'__behavioral.txt')
        file = open(filename,'r+')
        for i,line in enumerate(behData):
            line2 = [i] + line
            file.write(line2)
