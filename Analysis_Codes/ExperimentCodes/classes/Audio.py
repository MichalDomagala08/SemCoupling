import subprocess,sys
subprocess.check_call([sys.executable, "-m","pip","install","pyaudio"])

import pyaudio
import pygame
import wave
import threading
import time
from random import uniform
import sys,os

    
import time
from classes.recallTrial import recallTrial  # your existing class

class audioTrial:
    """
        Class that manages audio Playing and trials
        Consists of several methd
    
    """

    def __init__(self, audioPath,storyTimeDict, font, screen,currentStage,nextStage, output_control,entity=0,chunk_size=256,verbose=0,practice=False):

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
        self.practice = practice           # Whether we treat it as a practice run (No Logging of Information)
        self.currentStage = currentStage;  # Current Name of Stage
        self.nextStage = nextStage         # What Kind of stage do you expect next  
        self.storyParts = storyTimeDict["partNames"]
        self.entities = [storyTimeDict["partTimes"][0], storyTimeDict["partTimes"][0]] # Currently Tracked Entity
        self.state = "audio"  # Initial state

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
        if self.practice == False:
            self.output_control.write(f"       Value Logged {key}  at: {self.pausedTime - self.initialTime:.3f}; Reaction Time: {(self.pausedTime - self.probeOnset):.3f}")

            self.output_control.writeToEyeLink(message=f"\tKEYPRESS\t{key}\t0")              # Write KeyPress to Eyelink                  
            self.timingLog.append(["KEYPRESS",self.pausedTime  - self.initialTime ,(self.pausedTime - self.probeOnset),self.actual_duration])
            self.responsesTiming.append((key , self.pausedTime  - self.initialTime))
            
    def log_part_beg(self):
         if self.practice == False:
            self.timingLog.append(["BEGIN",(time.perf_counter() - self.initialTime),0,0])              
            self.responsesTiming.append(("BEGIN", (time.perf_counter() - self.initialTime)))

            message = f"\n   Beginning Part: {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f}";
            self.output_control.write(message)

            self.output_control.writeToEyeLink(message=f"\tPART\t{self.storyParts[self.currentStoryPart]:12}\tBEG")

    def log_part_end(self):
        if self.practice == False:

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
            #self.output_control.write(f"        Actual playback duration: {self.actual_duration:.3f} sec")

        # Lof that the audio is beginning:

        # Play audio on a separate thread
        threading.Thread(target=stream_audio, daemon=True).start()




    
    class Recording(recallTrial):
        """
        Subclass of recallTrial that:
        - Custom welcome screen without experimenter message
        - Uses super().display_message for multi-line prompts
        - Overlays a timer line turning red after 20s
        - Auto-stops at 25s without assurance prompt
        - Displays 'Nagranie zapisane!' for 1s after saving
        - Exposes thoughtContentEstimation() → True when done
        """
        def __init__(self, filename, font, screen, output_control, verbose=1):
            super().__init__(
                filename=filename,
                font=font,
                screen=screen,
                currentStage=None,
                nextStage=None,
                output_control=output_control,
                firstEntityName="",
                verbose=verbose
            )
            self._done = False
            self.start_time = None
            self._should_save = False
            self.timer_threshold = 20.0    # seconds to turn red
            self.record_duration = 25.0    # max recording length

        def thoughtContentEstimation(self):
            """
            Call each frame from audioTrial.run().
            Returns True once recording finished and saved.
            """
            events = pygame.event.get()

            # 1) Welcome screen
            if self.welcome_screen:
                lines = [
                    "Opowiedz przebieg swoich myśli od ostatniego przerwania historii.",
                    "",
                    "Naciśnij SPACJĘ, aby rozpocząć nagranie"
                ]
                super().display_message(lines)
                for evt in events:
                    if evt.type == pygame.KEYDOWN and evt.key == pygame.K_SPACE:

                        self.welcome_screen = False
                        self.start_time = time.time()
                        super().start_recording()
                return False

            # 2) Recording in progress
            elapsed = time.time() - self.start_time

            # Draw prompt via parent
            lines = [
                "Nagrywanie...",
                "",
                "Naciśnij ENTER, aby zakończyć nagrywanie"
            ]
            self.display_message(lines)

            # Overlay timer on second line
            line_height = 40
            total_height = len(lines) * line_height
            y0 = (self.screen.get_height() - total_height) // 2
            color = (255, 0, 0) if elapsed >= self.timer_threshold else (255, 255, 255)
            timer_text = f"{elapsed:.1f}s"
            timer_surf = self.font.render(timer_text, True, color)
            tx = (self.screen.get_width() - timer_surf.get_width()) // 2
            ty = y0 + line_height
            self.screen.blit(timer_surf, (tx+self.screen.get_width()*0.3, ty-self.screen.get_height()*0.3))
            pygame.display.flip()

            # Check stop conditions
            for evt in events:
                if (evt.type == pygame.KEYDOWN and
                    evt.key == pygame.K_RETURN and
                    self.recording):
                    self._should_save = True

            if elapsed >= self.record_duration and self.recording:
                self._should_save = True

            # Save and confirm once
            if self._should_save:
                super().stop_recording()
                super().save_audio()
                self._draw_and_flip(["Nagranie zapisane!"])
                pygame.display.flip()
                pygame.time.delay(1000)
                self._done = True
                return True

            return False

        # Delegate unchanged methods

        def _draw_and_flip(self, lines, line_height=40):
            """
            Helper to clear screen, display lines, and flip.
            """
            self.screen.fill((128, 128, 128))
            super().display_message(lines, line_height)
            pygame.display.flip()
        def start_recording(self):
            super().start_recording()

        def stop_recording(self):
            super().stop_recording()

        def save_audio(self):
            super().save_audio()

        def display_message(self, text_lines, line_height=40):
            super().display_message(text_lines, line_height)

    def run(self):
        """
        Runs the core of our audio trial.
        """

        # ==== Auditory Part of a Trial =====
        if self.state == "audio":

            if self.initFlag: # Initailise Experiment Begin as well as first Time to Pause
                self.initialTime = time.perf_counter()  # Start the timer
                self.pausedTime = self.initialTime
                self.initFlag = False;
            
            if self.trialFlag: # Startingto play The first sound at the beggining 
                self.play_audio()
                self.trialFlag = False

   
            self.display_fixation_cross()
            pygame.mouse.set_visible(False)
            
            # Drain/handle events safely to avoid crashes when participant clicks during playback
            # Quickly drop mouse events, then check only for quit/escape.
            pygame.event.clear((pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION))
            for evt in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
                if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                    try:
                        if self.stream:
                            self.stream.stop_stream()
                            self.stream.close()
                    except Exception:
                        pass
                    return "exit"

            # Check if the audio has finished
            if not self.audio_playing:
                self.probeOnset =  time.perf_counter();
                self.log_part_end()
                self.state = "probe"
                pygame.mouse.set_visible(True)


        # ==== Thought Probe Part of a Trial =====
        elif self.state == "probe":

            self.display_linear_scale_probe()

            ### Waiting for KeyPress on Thought Probe
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.stream.stop_stream()
                    self.stream.close()
                    return "exit"

                if self.get_linear_scale_value(): # If there was a correct button press 
                    #if self.paused_for_input and event.type == pygame.KEYDOWN:
                    #    if event.key == pygame.K_1 or event.key == pygame.K_2 or event.key == pygame.K_3 or event.key == pygame.K_4:
                    
                    currentKey = self.attention_value
                    self.log_keypress(currentKey) # Logs attention state value and moves on 
                        
                    pygame.mouse.set_visible(False)
                    self.currentRecorder = self.Recording(f"recall_{self.storyParts[self.currentStoryPart]}.wav", self.font,self.screen, self.output_control, verbose=self.verbose) # Create a Recorder instance 

                    self.state = "recall" # Change state to recall

        # --- 3. RECALL (RECORDING) STATE ---
        elif self.state == "recall":

                done = self.currentRecorder.thoughtContentEstimation()
                
                if done:
                    self.output_control.write(f"        PART {self.storyParts[self.currentStoryPart]:12}: Thought Recording Ended:   (Duration: {time.time() - self.initialTime})")

                    # flash “saved” message for 1 s
                    self.currentRecorder.display_message(["Nagranie zapisane!"])
                    pygame.time.delay(1000)

                    # set up for the next audio trial
                    self.currentStoryPart += 1
                    self.trialFlag = True
                    self.audio_playing = True
                    self.attention_value = None

                    self.state = "audio"
                    pygame.mouse.set_visible(False)



        # ─── 4) END‑OF‑BLOCK CHECK ─────────────────────────────────────────────────────
        if self.currentStoryPart >= len(self.audio_files):
            print(self.nextStage,self.currentStage)
            self.output_control.write("\nStory Ends.")
            return self.nextStage
        return self.currentStage

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



### OUR NEWEST EXPERIMENT:
storyTimeDictTest = {
    'partNames': [ 'KASIA_1', 'JANEK_1', 'KASIA_2',],
    'partTimes': [ 61.296, 59.525, 56.655 ]
}


storyTimeDict3 = {'partNames': [
    'KAROLINA_1', 'JANEK_1',  'KAROLINA_2', 'JANEK_2',  'KAROLINA_3', 'JANEK_3',  'KAROLINA_4', 'JANEK_4',  'KAROLINA_5', 'JANEK_5',  'KAROLINA_6','JANEK_6',  'KAROLINA_7', 'JANEK_7',  'KAROLINA_8', 'JANEK_8',  'KAROLINA_9', 'JANEK_9',
  'KAROLINA_10', 'JANEK_10',  'KAROLINA_11','JANEK_11',  'KAROLINA_12','JANEK_12',  'KAROLINA_13', 'JANEK_13',  'KAROLINA_14', 'JANEK_14',  'KAROLINA_15',
  'JANEK_15',  'KAROLINA_16','JANEK_16',  'KAROLINA_17', 'JANEK_17',  'KAROLINA_18',  'JANEK_18','KAROLINA_19',  'JANEK_19',  'KAROLINA_20', 'JANEK_20'],
 'partTimes': [42.553,  48.271, 40.601,  42.5, 43.513,  44.164,  43.247,  43.375,  46.17,  49.967, 42.457,  47.333,  48.069,  45.562,
  44.131,  40.003,  44.921,  44.452,  47.087,  45.145,  41.155,  42.415,  47.428,  47.183,  45.007,  43.833,
  47.652,  48.26,  49.082,  47.791,  44.025,  47.279,  47.141,  44.271,  42.404,  46.661,  40.963,  44.207,  44.591, 53.241]}


storyTimeDict3a = {'partNames': [
    'KAROLINA_1', 'JANEK_1',  'KAROLINA_2', 'JANEK_2',  'KAROLINA_3', 'JANEK_3',  'KAROLINA_4', 'JANEK_4',  'KAROLINA_5', 'JANEK_5',  'KAROLINA_6','JANEK_6',  'KAROLINA_7', 'JANEK_7',  'KAROLINA_8', 'JANEK_8',  'KAROLINA_9', 'JANEK_9',
  'KAROLINA_10', 'JANEK_10'],
 'partTimes': [42.553,  48.271, 40.601,  42.5, 43.513,  44.164,  43.247,  43.375,  46.17,  49.967, 42.457,  47.333,  48.069,  45.562,
  44.131,  40.003,  44.921,  44.452,  47.087,  45.145,  ]}

storyTimeDict3b = {'partNames': [
  'KAROLINA_11','JANEK_11',  'KAROLINA_12','JANEK_12',  'KAROLINA_13', 'JANEK_13',  'KAROLINA_14', 'JANEK_14',  'KAROLINA_15',
  'JANEK_15',  'KAROLINA_16','JANEK_16',  'KAROLINA_17', 'JANEK_17',  'KAROLINA_18',  'JANEK_18','KAROLINA_19',  'JANEK_19',  'KAROLINA_20', 'JANEK_20'],
 'partTimes': [ 41.155,  42.415,  47.428,  47.183,  45.007,  43.833,
  47.652,  48.26,  49.082,  47.791,  44.025,  47.279,  47.141,  44.271,  42.404,  46.661,  40.963,  44.207,  44.591, 53.241]}



storyTimeDict4 = {'partNames': ['KAROLINA_1',  'JANEK_1',  'KAROLINA_2',  'JANEK_2',  'KAROLINA_3',  'JANEK_3',  'KAROLINA_4',  'JANEK_4',  'KAROLINA_5',  'JANEK_5',  'KAROLINA_6',  'JANEK_6',  'KAROLINA_7',  'JANEK_7',  'KAROLINA_8',  'JANEK_8',  'KAROLINA_9',  'JANEK_9',  'KAROLINA_10',  'JANEK_10',  
                                'KAROLINA_11', 'JANEK_11', 'KAROLINA_12', 'JANEK_12', 'KAROLINA_13', 'JANEK_13', 'KAROLINA_14', 'JANEK_14', 'KAROLINA_15', 'JANEK_15', 'KAROLINA_16', 'JANEK_16', 'KAROLINA_17', 'JANEK_17', 'KAROLINA_18', 'JANEK_18', 'KAROLINA_19', 'JANEK_19', 'KAROLINA_20',  'JANEK_20'],
                  'partTimes': [40.313, 47.001, 39.491, 40.314, 37.935, 41.765, 40.740,  39.973,  41.999,  44.506,  41.475,  46.426,  46.746,  46.971,  42.734, 37.157, 45.977,  45.551,  47.313,  42.362, 
                                47.813, 42.502, 46.437, 41.989, 43.313, 46.662, 52.221, 50.919,  51.239,  50.193,  46.557,  52.967,  46.714,  46.309,  40.165,  49.049, 44.452, 43.759,  46.438,  53.681]}


storyTimeDict4a = {'partNames': ['KAROLINA_1',  'JANEK_1',  'KAROLINA_2',  'JANEK_2',  'KAROLINA_3',  'JANEK_3',  'KAROLINA_4',  'JANEK_4',  'KAROLINA_5',  'JANEK_5', 
                                  'KAROLINA_6',  'JANEK_6',  'KAROLINA_7',  'JANEK_7',  'KAROLINA_8',  'JANEK_8',  'KAROLINA_9',  'JANEK_9',  'KAROLINA_10',  'JANEK_10'],
                   'partTimes': [40.313, 47.001, 39.491, 40.314, 37.935, 41.765, 40.740,  39.973,  41.999,  44.506, 
                                 41.475, 46.426, 46.746, 46.971, 42.734, 37.157, 45.977,  45.551,  47.313,  42.362 ]}

storyTimeDict4b = {'partNames': [ 'KAROLINA_11', 'JANEK_11', 'KAROLINA_12', 'JANEK_12', 'KAROLINA_13', 'JANEK_13', 'KAROLINA_14', 'JANEK_14', 'KAROLINA_15', 'JANEK_15',
                                  'KAROLINA_16', 'JANEK_16', 'KAROLINA_17', 'JANEK_17', 'KAROLINA_18', 'JANEK_18', 'KAROLINA_19', 'JANEK_19', 'KAROLINA_20',  'JANEK_20'],
                   'partTimes': [  47.813, 42.502, 46.437, 41.989, 43.313, 46.662, 52.221, 50.919,  51.239,  50.193, 
                                   46.557, 52.967, 46.714, 46.309, 40.165, 49.049, 44.452, 43.759,  46.438,  53.681]}





Question =  ["Na chwilę obecną jaki jest twój stan uwagowy?",
             "Wciśnij odpowiedni przycisk:",
                    " ",
                    "%1 - Jeśli jesteś całkowicie zaangażowany w słuchaną historię. %",
                    "%2 - Jeśli myślisz o Historii ale jej nie słuchasz %",
                    "%3 - Jeśli myślisz o czymś zupełnie innym niż Historia %",
                    "%4 - Jeśli nie jesteś w stanie prześledzić swojego toku myśli %",]
