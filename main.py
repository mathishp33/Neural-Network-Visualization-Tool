import numpy as np
import math
import pygame as pg
import time

RES = WIDTH, HEIGHT = 1500, 750
screen = pg.display.set_mode(RES)
clock = pg.time.Clock()
pg.display.set_caption('NNVT : Neural Network Visualisation Tool')
mouse_pos = (0, 0)
L_BLUE = (50, 50, 200)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
D_GREEN = (0, 100, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
pg.font.init()
sf20 = pg.font.SysFont('Corbel', 20)
sf30 = pg.font.SysFont('Corbel', 30)
sf50 = pg.font.SysFont('Corbel', 50)
sf80 = pg.font.SysFont('Corbel', 80)
rects = []
neurons = []
connections = []
mouse_click = False
keys = pg.key.get_pressed()

class Neuron():
    def __init__(self, x, y):
        self.function = 'no'
        self.texts = ['no', 'relu', 'tanh', 'sigmoid']
        self.rects = []
        self.index = 0
        running = True
        for text in self.texts:
            self.text = sf30.render(text, True, WHITE, (100, 100, 100))
            self.rects.append(self.text.get_rect(center=(x, y+self.index)))
            self.index += 30
            screen.blit(self.text, self.rects[len(self.rects)-1])
        
        pg.display.update(self.rects)
        while running:
            mouse_pos = pg.mouse.get_pos()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.MOUSEBUTTONDOWN:
                    for rect in range(len(self.rects)):
                        if self.rects[rect].collidepoint(mouse_pos):
                            self.function = self.texts[rect]
                            running = False
            clock.tick(60)            
        self.type = 0
        self.x = x
        self.y = y
        self.value = 0
        self.output = 0
        self.entries = [0]
        self.text = sf30.render(str(self.value), True, WHITE)
    def update(self):
        self.value = sum(self.entries)
        if self.function == 'no':
            self.output = self.value
        if self.function == 'relu':
            self.output = self.value*(self.value>0)
        if self.function == 'tanh':
            self.output = np.tanh(self.value)
        if self.function == 'sigmoid':
            self.output == (np.tanh(self.value)+1)/2
        pg.draw.circle(screen, L_BLUE, (self.x, self.y), 50)
        self.rect = pg.draw.circle(screen, (100, 100, 100), (self.x, self.y), 50)
        self.text = sf30.render(str(round(self.output*10)/10), True, WHITE)
        self.text_rect = self.text.get_rect(center=(self.x , self.y))
        screen.blit(self.text, self.text_rect)
        self.entries = [0]


class InputNeuron():
    def __init__(self, x, y):
        self.type = 1
        self.x = x
        self.y = y
        self.sf = pg.font.SysFont('Corbel', 30)
        self.input = '0'
        self.value = 0
        self.text = self.sf.render(str(self.input), True, WHITE)
    def update(self):
        self.value = float(self.input)
        pg.draw.circle(screen, GREEN, (self.x, self.y), 50)
        self.rect = pg.draw.circle(screen, GREEN, (self.x, self.y), 50)
        self.text = self.sf.render(self.input, True, WHITE)
        self.text_rect = self.text.get_rect(center=(self.x , self.y))
        screen.blit(self.text, self.text_rect)
        if mouse_click and pg.draw.circle(screen, L_BLUE, (self.x, self.y), 50).collidepoint(mouse_pos):
            running = True
            while running:
                pg.draw.rect(screen, GREEN, pg.Rect(self.x-30, self.y-17, 70, 17*2))
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_BACKSPACE:
                            self.input = self.input[:-1]
                        else:
                            try : self.input += event.unicode
                            except: pass
                        if event.key == pg.K_RETURN:
                            running = False
                self.text = self.sf.render(self.input, True, YELLOW)
                self.text_rect = self.text.get_rect(center=(self.x , self.y))
                screen.blit(self.text, self.text_rect)
                pg.display.update(pg.Rect(self.x-30, self.y-17, 70, 17*2))
                clock.tick(60)
        
class OutputNeuron():
    def __init__(self, x, y):
        self.type = 2
        self.x = x
        self.y = y
        self.sf = pg.font.SysFont('Corbel', 30)
        self.value = 0
        self.entries = [0]
        self.text = self.sf.render(str(round(self.value*10)/10), True, WHITE)
    def update(self):
        self.value = sum(self.entries)
        pg.draw.circle(screen, RED, (self.x, self.y), 50)
        self.rect = pg.draw.circle(screen, RED, (self.x, self.y), 50)
        self.text = self.sf.render(str(round(self.value*10)/10), True, WHITE)
        self.text_rect = self.text.get_rect(center=(self.x , self.y))
        screen.blit(self.text, self.text_rect)
        self.entries = [0]
        
class Connection():
    def __init__(self, arg1):
        self.n1 = neurons[arg1[0][0]]
        self.n2 = neurons[arg1[0][1]]
        self.weight = arg1[1][0]
        self.bias = arg1[1][1]
        self.pos1 = self.n1.x, self.n1.y
        self.pos2 = self.n2.x, self.n2.y
        self.color = WHITE
        self.rect = pg.draw.line(screen, self.color, self.pos1, self.pos2, 5)
    def update(self):
        self.pos1 = self.n1.x, self.n1.y
        self.pos2 = self.n2.x, self.n2.y
        if self.n2.type != 1:
            self.n2.entries.append(self.n1.value*self.weight + self.bias)
        self.rect = pg.draw.line(screen, self.color, self.pos1, self.pos2, 5)
        pg.draw.line(screen, self.color, self.pos1, self.pos2, 5)
        
        
def bar_update():
    rects = []
    pg.draw.rect(screen, D_GREEN, pg.Rect(0, HEIGHT-70, WIDTH, 70))
    rects.append(pg.Rect(0, HEIGHT-100, WIDTH, 100))
    text0 = sf50.render('Neurons', True, BLACK, (0, 200, 0))
    text1 = sf50.render('Connections', True, BLACK, (200, 0, 0))
    text2 = sf50.render('Remove', True, BLACK, (135, 206, 235))
    rects.append(text0.get_rect(center=(90, HEIGHT-35)))
    rects.append(text1.get_rect(center=(320, HEIGHT-35)))
    rects.append(text2.get_rect(center=(550, HEIGHT-35)))
    screen.blit(text0, rects[1])
    screen.blit(text1, rects[2])
    screen.blit(text2, rects[3])
    return rects

def create_button():
    running = True
    rect = []
    text0 = sf30.render('Hidden Neuron', True, WHITE, (100, 100, 100))
    rect.append(text0.get_rect(center=(100, HEIGHT-100)))
    text1 = sf30.render('Input Neuron', True, WHITE, (0, 200, 0))
    rect.append(text1.get_rect(center=(100, HEIGHT-150)))
    text2 = sf30.render('Output Neuron', True, WHITE, (200, 0, 0))
    rect.append(text2.get_rect(center=(100, HEIGHT-200)))
    screen.blit(text0, rect[0])
    screen.blit(text1, rect[1])
    screen.blit(text2, rect[2])
    pg.display.update(rect)
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.MOUSEBUTTONDOWN:
                if rects[1].collidepoint(mouse_pos):
                    return 'canceled'
                if rect[0].collidepoint(mouse_pos):
                    return sub_create_button(0, rect)
                if rect[1].collidepoint(mouse_pos):
                    return sub_create_button(1, rect)
                if rect[2].collidepoint(mouse_pos):
                    return sub_create_button(2, rect)
                return False
        clock.tick(60)
        
def connections_button():
    running = True
    rect = []
    text0 = sf30.render('Create Connection', True, WHITE, (100, 100, 100))
    rect.append(text0.get_rect(center=(320, HEIGHT-100)))
    screen.blit(text0, rect[0])
    pg.display.update(rect)
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.MOUSEBUTTONDOWN:
                if rects[2].collidepoint(mouse_pos):
                    return 'canceled'
                if rect[0].collidepoint(mouse_pos):
                    return sub_connections(rect[0])
                return False
        clock.tick(60)
        
def sub_create_button(typeofneuron, rect):
    running = True
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT: return True
            if event.type == pg.MOUSEBUTTONDOWN:
                for rect1 in range(len(rect)):
                    if rect[rect1].collidepoint(mouse_pos) and typeofneuron != rect1:  
                        print("000")
                        return 'canceled'
                for rect in rects:
                    if rect.collidepoint(mouse_pos):
                        print('1111')
                        return 'canceled'
                return mouse_pos[0], mouse_pos[1], typeofneuron
        clock.tick(60)
        
def input_boxes(n1, n2):
    mid_point = ((n1.x+n2.x)/2, 
            (n1.y+n2.y)/2)
    parameters = []
    inputs = ['Weight : ', 'Bias']
    for i in inputs:
        running = True
        entry = ''
        while running:
            pg.draw.rect(screen, WHITE, pg.Rect(mid_point[0], mid_point[1], 100, 60))
            rect = pg.draw.rect(screen, WHITE, pg.Rect(mid_point[0], mid_point[1], 100, 60))
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_BACKSPACE:
                        entry = entry[:-1]
                    else:
                        try : entry += event.unicode
                        except: pass
                    if event.key == pg.K_RETURN:
                        running = False
            text = sf20.render(entry, True, WHITE, BLACK)
            text_rect = text.get_rect(center=(mid_point[0]+40 , mid_point[1]+36))
            text1 = sf20.render(i, True, BLACK)
            text1_rect = text1.get_rect(center=(mid_point[0]+40, mid_point[1]+20))
            screen.blit(text, text_rect)
            screen.blit(text1, text1_rect)
            pg.display.update(rect)
            clock.tick(60)
        try: parameters.append(float(entry))
        except: parameters.append(0)
    return parameters

def sub_connections(cancel_rect):
    output = [0, 0]
    count = 0
    running = True
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT: return True
            if event.type == pg.MOUSEBUTTONDOWN:
                if cancel_rect.collidepoint(mouse_pos):
                    return 'canceled'
                for rect in rects:
                    if rect.collidepoint(mouse_pos):
                        return 'canceled'
                for neuron in range(len(neurons)):
                    if neurons[neuron].rect.collidepoint(mouse_pos):
                        output[count] = neuron
                        if count == 1:
                            parameters = input_boxes(neurons[output[0]], neurons[output[1]])
                            return output, parameters
                        count += 1
                        time.sleep(0.2)
        clock.tick(60)

def sub_remove(element):
    running = True
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.MOUSEBUTTONDOWN:
                if element == 1:
                    for item in range(len(neurons)):
                        if neurons[item].rect.collidepoint(mouse_pos):
                            return element, item
                else:
                    for item in range(len(connections)):
                        if connections[item].rect.collidepoint(mouse_pos):
                            return element, item
                return 'canceled'
        clock.tick(60)
def remove_button():
    running = True
    rect = []
    text0 = sf30.render('Remove Connection', True, WHITE, (255, 0, 0))
    text1 = sf30.render('Remove Neuron', True, WHITE, (0, 255, 0))
    rect.append(text0.get_rect(center=(550, HEIGHT-100)))
    rect.append(text1.get_rect(center=(550, HEIGHT-150)))
    screen.blit(text0, rect[0])
    screen.blit(text1, rect[1])
    pg.display.update(rect)
    while running:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.MOUSEBUTTONDOWN:
                if rects[3].collidepoint(mouse_pos):
                    return 'canceled'
                if rect[0].collidepoint(mouse_pos):
                    return sub_remove(0)
                if rect[1].collidepoint(mouse_pos):
                    return sub_remove(1)
        clock.tick(60)

running = True
while running:
    screen.fill(BLACK)
    mouse_click = False
    keys = pg.key.get_pressed()
    mouse_pos = pg.mouse.get_pos()
    rects = bar_update()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            mouse_click = True
            if rects[1].collidepoint(mouse_pos):
                output = create_button()
                if output != 'canceled':
                    if output == True: 
                        running = False
                    else:
                        if output[2] == 0:
                            neurons.append(Neuron(output[0], output[1]))
                        elif output[2] == 1:
                            neurons.append(InputNeuron(output[0], output[1]))
                        elif output[2] == 2:
                            neurons.append(OutputNeuron(output[0], output[1]))
            if rects[2].collidepoint(mouse_pos):
                output = connections_button()
                if output != 'canceled':
                    if output == True:
                        running = False
                    else:
                        connections.append(Connection(output))
            if rects[3].collidepoint(mouse_pos):
                output = remove_button()
                if output != 'canceled':
                    if output == True:
                        running = False
                    else:
                        if output[0] == 0:
                            connections.pop(output[1])
                        else:
                            neurons.pop(output[1])
    if len(connections) != 0:
        for i in connections:
            i.update()
    if len(neurons) != 0:
        for j in neurons:
            j.update()
    pg.display.flip()
    clock.tick(60)
pg.quit()
