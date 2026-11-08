import subprocess,sys
subprocess.check_call([sys.executable, "-m","pip","install","pyaudio"])

import pyaudio
import pygame
import wave
import threading
import time
from random import uniform
import sys,os

class audioTrial:

    def __init__(self, audioPath,storyTimeDict, font, screen,currentStage,nextStage, output_control,entity=0,chunk_size=256,verbose=0):

        ### Screen
        self.font = font
        self.crossFont = pygame.font.Font(None, 100)

        self.screen = screen
        self.width = screen.get_size()[0]
        self.height = screen.get_size()[1]

        ### Audio Stream
        self.audio_folder = audioPath
        self.audio_files = self.load_audio_files()
        self.p = pyaudio.PyAudio()
        self.currentStoryPart = 0; # track which story segment is playing
        self.chunk_size = chunk_size
        self.stream = None  # Stream will be opened dynamically
        self.actual_duration = None # The Duration of Playing of a song will be stored Here 

        ### Audio Contoll Variables
        self.paused_position = 0        # Position in which we paused
        self.audio_playing = False      # Flag of Playing Auduo
        self.paused_for_input = False   # Flag of Pausing for input
        self.initialTime = 0            # intial time of starting our Audio
        self.pausedTime = 0             # Experiment Time in which we paused our stream
        self.nextProbeTime = 0          # Timing of a next probe
        self.initFlag = True            # Flag for initialization
        self.trialFlag = True           # Flag For measuring trials

        ### Debug and Return variables
        self.timingLog = []
        self.responsesTiming = []
        self.verbose = verbose
        self.output_control = output_control;
        sys.stdout = self.output_control  # Redirect standard output
        self.attention_value = None  # Initialize attention value
        self.buttonPressed = False  # Initialize button pressed state

        ### Staging and Experiment Mode:
        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next  
        self.storyParts = storyTimeDict["partNames"]
        self.entities = [storyTimeDict["partTimes"][0], storyTimeDict["partTimes"][0]]

    def load_audio_files(self):
            """ Load all WAV file as a path from the folder and sort them numerically. """
            files = [f for f in os.listdir(self.audio_folder) if f.endswith(".wav")]
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Sort by segment number
            return [os.path.join(self.audio_folder, f) for f in files]
    
    def log_keypress(self,key):
            
        """    if key == pygame.K_1:
            keyname = "1"
        elif key == pygame.K_2:
            keyname = "2"
        elif key == pygame.K_3:
            keyname = "3"
        elif key == pygame.K_4:
            keyname = "4"
        else:
            keyname = "d"

            self.output_control.write(f"WARNING SOMETHING ELSE WAS PRESSED {key}")
        if self.verbose:"""
            
        self.output_control.write(f"   Value Logged {key}  at: {self.pausedTime - self.initialTime:.3f}; Reaction Time: {(self.pausedTime - self.probeOnset):.3f}")

        self.output_control.writeToEyeLink(message=f"\tKEYPRESS\t{key}\t0")              # Write KeyPress to Eyelink                  
        self.timingLog.append(["KEYPRESS",self.pausedTime  - self.initialTime ,(self.pausedTime - self.probeOnset),self.actual_duration])
        self.responsesTiming.append((key , self.pausedTime  - self.initialTime))
            
    def log_part_beg(self):
        self.timingLog.append(["BEGIN",(time.perf_counter() - self.initialTime),0,0])              
        self.responsesTiming.append(("BEGIN", (time.perf_counter() - self.initialTime)))

        message = f"\n   Beginning Part: {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f}";
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"\tPART\t{self.storyParts[self.currentStoryPart]:12}\tBEG")

    def log_part_end(self):
        self.timingLog.append(["PROBE",(time.perf_counter() - self.initialTime),0,self.actual_duration])              
        self.responsesTiming.append(("PROBE", (time.perf_counter() - self.initialTime)))

        message = f"   Part ended:     {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f} (Duration: {self.actual_duration:.3f})"
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"\tPART\t{self.storyParts[self.currentStoryPart]:12}\tEND")
        # Write to Psychopy
    def display_fixation_cross(self):
        """
        Displays a '+' symbol in the center of the screen.
        """
        self.screen.fill((255 / 2, 255 / 2, 255 / 2))  # Gray background
        text_surface = self.crossFont.render("+", True, (255, 255, 255))  # White "+"
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def get_linear_scale_value(self):
        """
        Handles input logic for the linear scale probe.

        Updates:
        - self.attention_value: Percentage of the scale where the point is located (0 to 100).
        - self.buttonPressed: True if the "DALEJ" button is clicked.

        Returns:
        bool: True if the "DALEJ" button is clicked, False otherwise.
        """
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        # Check if the point is being dragged
        if mouse_pressed[0]:  # Left mouse button is held down
            if self.scale_x <= mouse_pos[0] <= self.scale_x + self.scale_width and \
           self.scale_y - self.point_radius <= mouse_pos[1] <= self.scale_y + self.scale_height + self.point_radius:
        
                self.point_x = mouse_pos[0]  # Update point position
                self.point_y = mouse_pos[1]  # Update point position
                # Calculate the attention value as a percentage (0 to 100)
                relative_position = (self.point_x - self.scale_x) / self.scale_width
                self.attention_value = int(relative_position * 100)

        # Check if the "DALEJ" button is clicked
            if self.button_x <= mouse_pos[0] <= self.button_x + self.button_width and \
            self.button_y <= mouse_pos[1] <= self.button_y + self.button_height:
                if self.attention_value is not None:  # Only allow clicking if the point has been moved
                    self.buttonPressed = True
                    return True

        return False
    def display_linear_scale_probe(self, scale_labels=["Brak rozproszenia","Pełne rozproszenie"], button_text="DALEJ"):
        """
        Draws the linear scale, draggable point, and "DALEJ" button.

        Parameters:
        scale_labels (list): A list of labels for the scale (e.g., ["Low", "High"]).
        button_text (str): Text for the confirmation button (default is "DALEJ").
        """
        # Colors
        background_color = (128, 128, 128)  # Gray
        scale_color = (255, 255, 255)       # White
        point_color = (0, 0, 0)             # Black
        button_color = (200, 200, 200)      # White
        button_hover_color = (175, 175, 175)  # Light gray for hover effect
        text_color = (255, 255, 255)              # Whire

        # Scale dimensions
        self.scale_width = self.width * 0.5
        self.scale_height = 10
        self.scale_x = (self.width -  self.scale_width) // 2
        self.scale_y = self.height // 2
        self.point_radius=10;

        # Button dimensions
        self.button_width = 150
        self.button_height = 50
        self.button_x = (self.width - self.button_width) // 2
        self.button_y = self.scale_y + 100
        
        # Draw the background
        self.screen.fill(background_color)

        # Draw the centered text
        center_text_surface = self.font.render("W jakim stopniu byłeś w tym momencie rozproszony podczas słuchania historii?", True, text_color)
        center_text_rect = center_text_surface.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(center_text_surface, center_text_rect)

        # Draw the reminder text
        center_text_surface2 = self.font.render("(Zaznacz myszką na poniższej skali w dowolnym miejscu)", True, text_color)
        center_text_rect2 = center_text_surface.get_rect(center=(self.width // 2, self.height // 3.3))
        self.screen.blit(center_text_surface2, center_text_rect2)

        # Draw the scale
        pygame.draw.rect(self.screen, scale_color, (self.scale_x, self.scale_y,  self.scale_width, self.scale_height))
        # Draw the draggable point
        if self.attention_value is not None:
            point_y = self.scale_y + self.scale_height // 2
            pygame.draw.circle(self.screen, point_color, (self.point_x, point_y), self.point_radius)

        # Draw the scale markings
        marking_positions = [
            self.scale_x,  # Start of the scale
            self.scale_x + self.scale_width // 2,  # Center of the scale
            self.scale_x + self.scale_width  # End of the scale
        ]
        for pos in marking_positions:
            pygame.draw.line(self.screen, (0,0,0), (pos, self.scale_y - 5), (pos, self.scale_y + self.scale_height + 5), 2)



        # Draw the scale labels
        for i, label in enumerate(scale_labels):
            label_surface = self.font.render(label, True, text_color)
            label_x = self.scale_x + i * ( self.scale_width // (len(scale_labels) - 1)) - label_surface.get_width() // 2
            label_y = self.scale_y - 50
            self.screen.blit(label_surface, (label_x, label_y))

        # Draw the button
        mouse_pos = pygame.mouse.get_pos()
        if self.button_x <= mouse_pos[0] <= self.button_x + self.button_width and self.button_y <= mouse_pos[1] <= self.button_y + self.button_height:
            pygame.draw.rect(self.screen, button_hover_color, (self.button_x, self.button_y, self.button_width, self.button_height))
        else:
            pygame.draw.rect(self.screen, button_color, (self.button_x, self.button_y, self.button_width, self.button_height))

        # Draw the button text
        button_surface = self.font.render(button_text, True, text_color)
        button_text_x = self.button_x + (self.button_width - button_surface.get_width()) // 2
        button_text_y = self.button_y + (self.button_height - button_surface.get_height()) // 2
        self.screen.blit(button_surface, (button_text_x, button_text_y))

        pygame.display.flip()


    def display_thought_probe(self,line_height=50):
        """
            Render multi-line text with dynamic left margin (in pixels) and special centering for lines with '# #'.
        """
        # Determine the width of the longest line in pixels
        max_line_pixel_width = max(self.font.size(line.replace("#", "").strip())[0] for line in Question)

        # Calculate the dynamic left margin as a proportion of the unused space
        margin = (self.width - max_line_pixel_width) // 2

        # Calculate the Y position to center the entire block vertically
        total_text_height = len(Question) * line_height
        y_offset = (self.height - total_text_height) // 2

        for i, line in enumerate(Question):
            # Check if the line should be centered (contains '# #')
            if line.startswith("#") and line.endswith("#"):
                clean_line = line.replace("#", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)

            elif line.startswith("%") and line.endswith("%"):
                clean_line = line.replace("%", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(topleft=(self.width // 3, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)
            else:
                # Left-aligned text at the dynamic margin
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (margin, y_offset + i * line_height))
        pygame.display.flip()

    def play_audio(self):
        """
        Starts/resumes playing the audio from each file
        """

        self.audio_playing = True
        self.paused_for_input = False  # Reset for next thought probe

        file_path = self.audio_files[self.currentStoryPart]
        self.audio_file = wave.open(file_path, 'rb')

        if self.stream:
            self.stream.close()  # Close previous stream if it exists

        # Open Stream for playing
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.audio_file.getsampwidth()),
            channels=self.audio_file.getnchannels(),
            rate=self.audio_file.getframerate(),
            output=True
        )
        self.log_part_beg()


        def set_volume(self, dB):
            """
            Set the volume in decibels.
            """
            self.volume = 10 ** (dB / 20)
            self.output_control.write(f"Volume set to {dB} dB (linear scale: {self.volume:.2f})")
        def stream_audio():
            start_time = time.perf_counter()  # Capture the actual start time
            while self.audio_playing:
                data = self.audio_file.readframes(self.chunk_size)
                if data == b'':  # End of file
                    self.audio_playing = False
                    break
                
                   # Adjust volume
               # audio_data = wave.struct.unpack("%dh" % (len(data) // 2), data)  # Unpack audio data
               # adjusted_data = [int(sample * self.volume) for sample in audio_data]  # Scale by volume
               # adjusted_data = wave.struct.pack("%dh" % len(adjusted_data), *adjusted_data)  # Pack back to bytes



                self.stream.write(data)
                # Track the current position (in samples)

            self.audio_playing = False
            self.actual_duration = time.perf_counter() - start_time  # Calculate actual time taken
            self.output_control.write(f"Actual playback duration: {self.actual_duration:.3f} sec")

        # Lof that the audio is beginning:

        # Play audio on a separate thread
        threading.Thread(target=stream_audio, daemon=True).start()

    
    def run(self):
        """
        Runs the core of our audio trial.
        """

        if self.initFlag: # Initailise Experiment Begin as well as first Time to Pause
            self.initialTime = time.perf_counter()  # Start the timer
            self.pausedTime = self.initialTime
            self.initFlag = False;
        
        if self.trialFlag: # Startingto play The first sound at the beggining 
            self.play_audio()
            self.trialFlag = False

        if self.paused_for_input:
            self.display_linear_scale_probe()
        else:
            self.display_fixation_cross()
        pygame.mouse.set_visible(self.paused_for_input)

        # Check if the audio has finished
        if not self.audio_playing:
            if not self.paused_for_input:
                self.probeOnset =  time.perf_counter();
                self.log_part_end()
            self.paused_for_input = True

        ### Waiting for KeyPress on Thought Probe
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.stream.stop_stream()
                self.stream.close()
                return "exit"

            if self.paused_for_input: 
                if self.get_linear_scale_value():
                #if self.paused_for_input and event.type == pygame.KEYDOWN:
                #    if event.key == pygame.K_1 or event.key == pygame.K_2 or event.key == pygame.K_3 or event.key == pygame.K_4:
                
                    self.paused_for_input = False
                    self.pausedTime = time.perf_counter()  # Reset timer
                    currentKey = self.attention_value
                    self.log_keypress(currentKey) # Logs KeyPresses
                        
                    self.currentStoryPart += 1
                    self.trialFlag = True
                    self.audio_playing = True
                    self.attention_value = None  # Reset attention value

        if self.currentStoryPart >= len(self.audio_files):
            self.output_control.write("\nStory Ends.")
            return self.nextStage  # No more segments → End experiment

        return  self.currentStage # Usually return story1 if audio has not ended or we hadnt exit

# A list of Timings of story 1: Used to Clock the Onset of Each Fragment: (in seconds)
storyTimeDict2 = data = {
    'partNames': [ 'ADAM_1', 'MARIA_1', 'ADAM_2', 'MARIA_2', 'ADAM_3', 'MARIA_3', 'ADAM_4', 'MARIA_4', 'ADAM_5', 'MARIA_5', 'ADAM_6', 'MARIA_6', 'ADAM_7', 
                  'MARIA_7', 'ADAM_8', 'MARIA_8', 'ADAM_9', 'MARIA_9', 'ADAM_10', 'MARIA_10', 'ADAM_11', 'MARIA_11'],
    'partTimes': [ 56.678, 56.688, 64.743, 59.355, 58.085, 56.379, 64.487, 62.353, 
                  61.563, 57.115, 59.493, 57.019, 56.283, 59.941, 57.679, 58.043, 59.707, 60.827, 59.663, 59.109, 60.443, 58.832, ]
}

storyTimeDict1 = {
    'partNames': [ 'KASIA_1', 'JANEK_1', 'KASIA_2', 'JANEK_2', 'KASIA_3', 'JANEK_3', 'KASIA_4', 'JANEK_4', 'KASIA_5', 'JANEK_5', 'KASIA_6', 'JANEK_6', 'KASIA_7', 
                  'JANEK_7', 'KASIA_8', 'JANEK_8', 'KASIA_9', 'JANEK_9', 'KASIA_10', 'JANEK_10',  'KASIA_11', 'JANEK_11'],
    'partTimes': [ 61.296, 59.525, 56.655, 59.92, 65.147, 57.574, 61.99, 65.329, 60.965, 
                  60.454, 57.947, 64.176, 60.955, 60.241, 59.472, 57.115, 59.387, 57.306, 58.693, 62.267, 58.063, 61.295, ]
}
storyTimeDict2c = data = {
    'partNames': ['INTRODUCTION', 'ADAM_1', 'MARIA_1', 'ADAM_2', 'MARIA_2', 'ADAM_3', 'MARIA_3', 'ADAM_4', 'MARIA_4', 'ADAM_5', 'MARIA_5', 'ADAM_6', 'MARIA_6', 'ADAM_7', 
                  'MARIA_7', 'ADAM_8', 'MARIA_8', 'ADAM_9', 'MARIA_9', 'ADAM_10', 'MARIA_10', 'ADAM_11', 'MARIA_11', 'ENDING'],
    'partTimes': [40.505, 56.678, 56.688, 64.743, 59.355, 58.085, 56.379, 64.487, 62.353, 
                  61.563, 57.115, 59.493, 57.019, 56.283, 59.941, 57.679, 58.043, 59.707, 60.827, 59.663, 59.109, 60.443, 58.832, 64.433]
}

storyTimeDict1c = {
    'partNames': ['INTRODUCTION', 'KASIA_1', 'JANEK_1', 'KASIA_2', 'JANEK_2', 'KASIA_3', 'JANEK_3', 'KASIA_4', 'JANEK_4', 'KASIA_5', 'JANEK_5', 'KASIA_6', 'JANEK_6', 'KASIA_7', 
                  'JANEK_7', 'KASIA_8', 'JANEK_8', 'KASIA_9', 'JANEK_9', 'KASIA_10', 'JANEK_10',  'KASIA_11', 'JANEK_11', 'ENDING'],
    'partTimes': [35.257, 61.296, 59.525, 56.655, 59.92, 65.147, 57.574, 61.99, 65.329, 60.965, 
                  60.454, 57.947, 64.176, 60.955, 60.241, 59.472, 57.115, 59.387, 57.306, 58.693, 62.267, 58.063, 61.295, 55.909]
}

storyTimeDict3 = {'partNames': [
    'KAROLINA_1', 'JANEK_1',  'KAROLINA_2', 'JANEK_2',  'KAROLINA_3', 'JANEK_3',  'KAROLINA_4', 'JANEK_4',  'KAROLINA_5', 'JANEK_5',  'KAROLINA_6','JANEK_6',  'KAROLINA_7', 'JANEK_7',  'KAROLINA_8', 'JANEK_8',  'KAROLINA_9', 'JANEK_9',
  'KAROLINA_10', 'JANEK_10',  'KAROLINA_11','JANEK_11',  'KAROLINA_12','JANEK_12',  'KAROLINA_13', 'JANEK_13',  'KAROLINA_14', 'JANEK_14',  'KAROLINA_15',
  'JANEK_15',  'KAROLINA_16','JANEK_16',  'KAROLINA_17', 'JANEK_17',  'KAROLINA_18',  'JANEK_18','KAROLINA_19',  'JANEK_19',  'KAROLINA_20', 'JANEK_20'],
 'partTimes': [42.553,  48.271, 40.601,  42.5, 43.513,  44.164,  43.247,  43.375,  46.17,  49.967, 42.457,  47.333,  48.069,  45.562,
  44.131,  40.003,  44.921,  44.452,  47.087,  45.145,  41.155,  42.415,  47.428,  47.183,  45.007,  43.833,
  47.652,  48.26,  49.082,  47.791,  44.025,  47.279,  47.141,  44.271,  42.404,  46.661,  40.963,  44.207,  44.591, 53.241]}
Question =  ["Na chwilę obecną jaki jest twój stan uwagowy?",
             "Wciśnij odpowiedni przycisk:",
                    " ",
                    "%1 - Jeśli jesteś całkowicie zaangażowany w słuchaną historię. %",
                    "%2 - Jeśli myślisz o Historii ale jej nie słuchasz %",
                    "%3 - Jeśli myślisz o czymś zupełnie innym niż Historia %",
                    "%4 - Jeśli nie jesteś w stanie prześledzić swojego toku myśli %",]
