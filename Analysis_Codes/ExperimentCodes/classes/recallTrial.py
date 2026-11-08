import pygame
import pyaudio
import wave
import threading
import sys
import os
import time

class recallTrial:

    def __init__(self,filename,font,screen,currentStage,nextStage,output_control ,firstEntityName,channels = 1 ,rate = 44100,chunk_size = 1024,verbose=1 ):


        # Screen setup
        self.width = screen.get_size()[0]
        self.height = screen.get_size()[1]
        self.screen = screen
        self.font = font
        self.firstEntityName = firstEntityName
        # Recording setup
        self.chunk_size = chunk_size  # Audio chunk size
        self.sample_format = pyaudio.paInt16  # 16-bit audio
        self.channels = channels  # Mono
        self.rate = rate  # Sampling rate (44.1 kHz)
        self.p = pyaudio.PyAudio()

        self.frames = []  # Store recorded audio data
        self.recording = False  # Flag to check if we are recording
        self.welcome_screen = True

        # Staging:
        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next 
        self.output_control = output_control;
        self.verbose = verbose
        self.assurance = False
        if self.verbose:
            sys.stdout = self.output_control  # Redirect standard output

        if not os.path.isdir(os.path.join(self.output_control.dataPath,self.output_control.ID ,"Recalls")):
            os.mkdir(os.path.join(self.output_control.dataPath,self.output_control.ID,"Recalls"))
        self.audio_filename = os.path.join(self.output_control.dataPath,self.output_control.ID,"Recalls",filename)
        self.initialTime = time.time();

    def start_recording(self):
        """
        Start recording audio on a separate thread.
        """
        self.frames = []  # Clear any previous recordings
        self.recording = True

        def record_audio():
            stream = self.p.open(format=self.sample_format,
                            channels=self.channels,
                            rate=self.rate,
                            frames_per_buffer=self.chunk_size,
                            input=True)

            while self.recording:
                data = stream.read(self.chunk_size)
                self.frames.append(data)

            # Stop and close the stream when recording ends
            stream.stop_stream()
            stream.close()

            # Save the recorded audio to a WAV file
            self.save_audio()

        # Start recording in a background thread
        threading.Thread(target=record_audio, daemon=True).start()


    def stop_recording(self):
        """
        Stop recording audio.
        """
        self.recording = False


    def save_audio(self):
        """
        Save recorded audio to a WAV file.
        """

        wf = wave.open(self.audio_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.output_control.write(f"        Recording saved as: {self.audio_filename}")


    def display_message(self,text_lines,line_height=40):
        """
        Display a message at the center of the screen.
        """

        # Determine the width of the longest line in pixels
        max_line_pixel_width = max(self.font.size(line.replace("#", "").strip())[0] for line in text_lines)

        # Calculate the dynamic left margin as a proportion of the unused space
        margin = (self.width - max_line_pixel_width) // 2

        # Calculate the Y position to center the entire block vertically
        total_text_height = len(text_lines) * line_height
        y_offset = (self.height - total_text_height) // 2

        for i, line in enumerate(text_lines):
            # Check if the line should be centered (contains '# #')
            if line.startswith("#") and line.endswith("#"):
                clean_line = line.replace("#", "").strip()  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)
            else:
                # Left-aligned text at the dynamic margin
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (margin, y_offset + i * line_height))

     

    def run(self):

        if self.welcome_screen:
            self.display_message(["Poczekaj teraz na eksperymentatora, aż ustawi sprzęt potrzebny do nagrywania.", " ", "Gdy wszystko będzie gotowe, naciśnij SPACJĘ by nagrać "," ustne odpamiętanie fragmentów historii dotyczących bohatera imieniem"," "," ",f"#{self.firstEntityName}#"])
        elif self.recording and self.assurance == False:
            self.display_message(["Nagrywanie... ","","Naciśniej ENTER by zakończyć nagrywanie "])
        elif self.recording and self.assurance:
            self.display_message(["Czy na pewno chcesz zakończyć nagrywanie?","", "Naciśnij ENTER by zakończyć nagrywanie "," lub SPACJĘ by kontynuować nagrywanie"])
        for event in pygame.event.get() :
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.p.terminate()
                return "exit"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.welcome_screen:
                    self.welcome_screen = False
                    self.output_control.write(f"Recording Started: (Duration: {time.time() - self.initialTime})")
                    self.start_recording()  # Start recording
                    self.welcome_screen = False
                elif event.key == pygame.K_RETURN and self.recording and not self.assurance:  # Press Enter to stop
                    self.assurance = True;
                elif event.key == pygame.K_SPACE and self.assurance:  # Press Enter to stop
                    self.assurance = False;
                elif event.key == pygame.K_RETURN and self.assurance and self.recording:  # Press Enter to stop
                    self.stop_recording()
                    self.display_message("Recording stopped. Saving...")
                    pygame.time.delay(2000)  # Small delay to show the message
                    self.output_control.write(f"Recording Ended:   (Duration: {time.time() - self.initialTime})")

                    return  self.nextStage 

        return  self.currentStage 

