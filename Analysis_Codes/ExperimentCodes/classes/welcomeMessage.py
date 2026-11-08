import pygame
from pygame.locals import * 


class welcomeMessage:
    """
    Class that governs the displaying of inter-trial messages to participants (those that are not contingent on Trials )
    """

    def __init__(self, textList,font,screen,currentStage,nextStage, msgCount = 0, marginAdjust = 0,):
        self.textList  = textList
        self.msgCount = msgCount
        self.font = font
        self.screen = screen
        self.height = screen.get_size()[1]
        self.width =  screen.get_size()[0]

        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next 


    def render_multiline_text(self,text_lines, line_height=50):
        """
        Render multi-line text with dynamic left margin (in pixels) and special centering for lines with '# #'.
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
                clean_line = line.replace("#", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)

            elif line.startswith("%") and line.endswith("%"):
                clean_line = line.replace("%", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(topleft=(self.width // 4, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)
            else:
                # Left-aligned text at the dynamic margin
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (margin, y_offset + i * line_height))

    def run(self):
        #self.screen.fill((255/2,255/2, 255/2))

        self.render_multiline_text(self.textList[self.msgCount])
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):  # Quit
                return "exit"

            if event.type == KEYDOWN and event.key == K_SPACE:  # Progress on SPACE press
                self.msgCount += 1
                

                if self.msgCount >= len(self.textList):
                    return  self.nextStage # return flag for the next experiment


        return  self.currentStage  # return flag for welcome period


def generateMessages(firstEntityName):


    WelcomeMessage1 =  [
                                "#Witaj w procedurze eksperymentalnej!#",
                        "",
                        "Procedura ma na celu zbadanie efektów podtrzymania uwagi w zadaniu słuchowym",
                        "",     
                        "Badanie jest dobrowolne, a wszystkie dane są anonimowe i poufne.",
                        "Możesz w każdej chwili wycofać się z badania bez podawania przyczyny.",
                        " ",
                                "#Naciśnij SPACJĘ, aby przejść dalej #"]
    
    WelcomeMessage11 = [ "Za chwilę usłyszysz historię trwającą ok. 40 min.",
                        "W trakcie jej słuchania, prosimy o utrzymanie wzroku",
                        "na krzyżyku wyświetlonym na ekranie.",
                        " ",
                        "Dodatkowo poprosimy Cię o zwracanie szczególnej uwagi na losy konkretnego bohatera.",
                        " ",
                        "Po odsłuchaniu, zostaniesz poproszony/a o ustne",
                        "odpamiętanie JEDYNIE fragmentów dotyczących tego bohatera.",
                        " ",
                                "#Naciśnij SPACJĘ aby przejść dalej #"]

    WelcomeMessage2 =  ["Historia będzie co jakis czas przerywana przez pytanie",
                        "wyświetlone na środku ekranu:", 
                        " ",
                                "#\"W jakim stopniu byłeś w tym momencie rozproszony podczas słuchania historii? \"#",
                        " ",
                        "W odpowiedzi, będziesz musiał/a myszką wybrać punkt na liniowej skali w taki sposób, ", 
                        "aby pokazywał twoją subiektywną ocenę rozproszenia od słuchania historii:",
                        " ",
                        "% Początek skali oznacza pełną uwagę na historii%",
                        "% Koniec skali oznacza rozproszenie od słuchania historii%",
                        " ",
                        "Nasz eksperyment ma na celu badanie uważności, nie obawiaj się więc raportować rozproszenia.",
                        "Postaraj się ocenić możliwie jak najbardziej szczerze swój stan uwagowy.",
                        "",
                                "#Naciśnij SPACJĘ aby przejść dalej#"] 
    
    WelcomeMessage2b =  ["Po udzieleniu odpowiedzi zostaniesz poproszony/a o przywołanie ustne treści", 
                         "swoich myśli od momentu ostatniego przerwania historii w trakcie jej słuchania",
                        " ",
                        "Po naciśnięciu spacji, rozpocznie się nagrywanie. Dodatkowo, na ekranie wyświetlany",
                         "będzie czas mówienia. Prosimy o ograniczenie się do wypowiedzi trwajcej 25 sekund.",
                         "W przypadku przekroczenia czasu, procedura będzie kontynowana, a dotychczasowe ",
                         "nagranie zapisane.",
                        "", 
                         "Nagranie jest całkowicie anonimowe. W przypadku prywatnych myśli, prosimy",
                         "o wspomnienie o nich na takim poziomie jaki nie przekracza Twojego komfortu.",
                         "",
                         "Celem odpamiętania jest próba oszacowania myśli nie związanych z zadaniem, ",
                         "więc nie obawiaj się o nich wspominać. Staraj się szczerze raportować swoje myśli,",
                        " Tak dokładnie, jak to jest tylko możliwe.",
                        "",
                                "#Naciśnij SPACJĘ aby przejść dalej#"] 
    
    WelcomeMessage2c = ["Teraz przejdziemy do próbnego eksperymentu, by lepiej zaznajomić Cię z procedurą.",
                        "Zostanie Ci przedstawiona historia niezwiązana z eksperymentem.",
                        "",
                        "Nałóż słuchawki i przygotuj się.",
                        "",
                        "Naciśnij SPACJĘ aby rozpocząć próbny eksperyment"]

    WelcomeMessage3 =  [f"Za chwilę przejdziesz do kalibracji okulografu.",
                        "Kieruj wzrok w stronę obiektów wyświetlanych na ekranie.",
                        "",
                        "#Naciśnij SPACJĘ aby rozpocząć kalibrację#"]


    WelcomeMessage21alt =  [f"Za chwilę usłyszysz właściwą historię. Nałóż słuchawki i przygotuj się.",
                        "Postaraj się skupić i zapamiętać wszystkie fragmenty dotyczące"
                        " bohatera imieniem:",
                        "",
                                            f"#{firstEntityName}#",
                        "",
                        "Pamiętaj, że po zakończeniu eksperymentu będziesz proszony o odpamiętanie",
                         " szczegółów dotyczących tylko tego bohatera.",
                         " ",
                        "#Naciśnij SPACJĘ aby zacząć słuchać historii#"]

    WelcomeMessage4 =  [f"Za chwilę przejdziesz do ponownej kalibracji okulografu.",
                        "Tym razem, procedura będzie kontynuowana automatycznie po jej zakończeniu.",
                        "Kieruj wzrok w stronę obiektów wyświetlanych na ekranie.",
                        "",
                        "#Naciśnij SPACJĘ aby rozpocząć kalibrację#"]
  
    exitMessage1 =          ["#Dziękujemy za udział w badaniu#",
                            "",
                            "Zapraszamy teraz do wypełnienia krótkiego",
                            "kwestionariusza"]

    """

      WelcomeMessage21 =  [f"Za chwilę usłyszysz pierwszą historię.",
                        "Postaraj się zapamiętać jak najwięcej szczegółów.",
                        "",
                        "#Naciśnij SPACJĘ aby zacząć słuchać historii#"]
    
    secondStoryMessage1 =  ["Odpocznij przez kilka minut.",
                            "",
                            "Gdy będziesz gotowy aby przejsć dalej,",
                            "Naciśnij SPACJĘ"]

    secondStoryMessage2 =  [f"Za chwilę przejdziesz do kalibracji okulografu.",
                        "Kieruj wzrok w stonę obiektów na ekranie",
                        "",
                        "#Wciśnij SPACJĘ aby rozpocząć kalibrację#"]

    secondStoryMessage3 =  [f"Za chwilę usłyszysz drugą historię.",
                            "Postaraj się skupić na wszystkich odniesieniach",
                            " do bohatera imieniem:",
                            "",
                                        f"#{secondEntityName}#",
                            "",
                            "#Naciśnij SPACJĘ aby zacząć słuchać historii#"]


    """
    
    return [WelcomeMessage1,WelcomeMessage11,WelcomeMessage2,WelcomeMessage2b,WelcomeMessage2c,WelcomeMessage3,WelcomeMessage21alt,WelcomeMessage4,exitMessage1]
