import pygame
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Inicializa o pygame e define as cores
pygame.init()
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Define a tela principal e áreas de desenho
screen_width, screen_height = 900, 450  # 600x450 pixels
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw and Recognize Digit")

# Define o retângulo do canvas (área de desenho) e o botão de limpar
canvas_rect = pygame.Rect(10, 10, 400, 400)
button_rect = pygame.Rect(420, 10, 150, 40)  # botão no lado direito

# Cria o canvas onde o usuário pode desenhar (fundo branco)
canvas = pygame.Surface(canvas_rect.size)
canvas.fill(WHITE)

# Carrega o modelo treinado – ajuste o nome do arquivo conforme necessário
model = load_model("TRAINED_MODEL.keras")
# O modelo deve esperar entrada com shape (28, 28, 1)

# Define variáveis para controle do desenho
drawing = False
last_pos = None

# Define fonte para exibição de textos
font = pygame.font.SysFont(None, 30)

#VARIAVEIS RELEVANTES
thickness=30

# Função para desenhar uma linha suave no canvas
def draw_line(surface, start, end, color=BLACK, thickness=thickness):
    pygame.draw.line(surface, color, start, end, thickness)


# Função para converter o canvas desenhado em uma imagem que a rede pode processar e para obter a predição

def center_drawing(gray_image, threshold=200):
    """
    Recebe uma imagem em escala de cinza (28x28) e centraliza o desenho.
    Valores de pixel abaixo do threshold são considerados parte do desenho.
    """
    # Cria uma máscara dos pixels "desenhados" (mais escuros que o threshold)
    mask = gray_image < threshold
    if np.sum(mask) == 0:
        # Se nada foi desenhado, retorna a imagem original
        return gray_image

    # Obtém as coordenadas onde o desenho existe
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 para incluir o último índice

    # Recorta a região desenhada
    cropped = gray_image[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape

    # Cria um canvas novo 28x28 com fundo branco (valor 255)
    new_canvas = np.ones((28, 28)) * 255

    # Calcula onde posicionar o recorte para centralizá-lo
    start_y = (28 - crop_h) // 2
    start_x = (28 - crop_w) // 2

    # Cola o desenho centralizado no novo canvas
    new_canvas[start_y:start_y + crop_h, start_x:start_x + crop_w] = cropped

    return new_canvas


def get_prediction(surface):
    # Redimensiona o canvas para 28x28 pixels
    small_image = pygame.transform.smoothscale(surface, (28, 28))
    # Obtem o array RGB (resultado com shape (largura, altura, 3))
    image_array = pygame.surfarray.array3d(small_image)
    # Transpõe para ter formato (altura, largura, canais)
    image_array = np.transpose(image_array, (1, 0, 2))
    # Converte a imagem para escala de cinza usando média ponderada (luminosidade)
    image_gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Centraliza o desenho (você pode incluir aqui sua função de centralização, se já implementada)
    centered_gray = center_drawing(image_gray)  # supondo que center_drawing já esteja definida

    # Normaliza a imagem para o intervalo [-1, 1]
    image_norm = (centered_gray.astype(np.float32) - 127.5) / 127.5
    # Ajusta para o formato esperado pelo modelo: (1, 28, 28, 1)
    image_norm = np.expand_dims(image_norm, axis=0)  # batch
    image_norm = np.expand_dims(image_norm, axis=-1)  # canal

    # Faz a predição com o modelo
    pred = model.predict(image_norm)
    predicted_digit = np.argmax(pred)

    # Para visualização, cria uma surface a partir da imagem centralizada
    # Primeiro, empilha a imagem em 3 canais para formar uma imagem RGB
    centered_rgb = np.stack([centered_gray, centered_gray, centered_gray], axis=2).astype(np.uint8)
    centered_surface = pygame.surfarray.make_surface(centered_rgb)

    # Corrige a rotação: se a imagem está rota 90 graus para a esquerda, rotaciona-a 90 graus para a direita.
    centered_surface = pygame.transform.rotate(centered_surface, 270)
    centered_surface = pygame.transform.flip(centered_surface, True, False)

    return predicted_digit, pred[0], centered_surface


# Configura o clock para controlar o FPS
clock = pygame.time.Clock()

# Loop principal
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Eventos do mouse
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botão esquerdo
                if button_rect.collidepoint(event.pos):
                    # Limpa o canvas se o botão "Clear" for clicado
                    canvas.fill(WHITE)
                elif canvas_rect.collidepoint(event.pos):
                    drawing = True
                    # Ajusta a posição relativa ao canvas
                    last_pos = (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y)
                if last_pos is not None:
                    draw_line(canvas, last_pos, current_pos)
                last_pos = current_pos

    # Obtenha a predição da rede para o desenho atual (em tempo real) e a imagem 28x28
    predicted_digit, probabilities, small_image = get_prediction(canvas)

    # Desenha o fundo da tela
    screen.fill(GRAY)

    # Exibe o canvas desenhado
    screen.blit(canvas, canvas_rect.topleft)

    # Desenha o botão de limpar
    pygame.draw.rect(screen, WHITE, button_rect)
    btn_text = font.render("Clear", True, BLACK)
    screen.blit(btn_text, (button_rect.x + 35, button_rect.y + 10))

    # Exibe a predição da rede
    result_text = font.render(f"Digit: {predicted_digit}", True, BLACK)
    screen.blit(result_text, (420, 70))

    # Exibe as probabilidades para cada dígito (opcional)
    prob_text = font.render("Probabilities:", True, BLACK)
    screen.blit(prob_text, (420, 110))
    for i, p in enumerate(probabilities):
        p_text = font.render(f"{i}: {p:.2f}", True, BLACK)
        screen.blit(p_text, (420, 140 + i * 25))

    # Mostra a imagem pixelada 28x28 ampliada para visualização
    # Primeiro, converte o small_image para um tamanho maior, por exemplo, 5x o tamanho original (28x28 -> 140x140)
    pixelated_image = pygame.transform.scale(small_image, (280, 280))
    # Blita essa imagem na tela (posição: 420, 300)
    screen.blit(pixelated_image, (600, 10))

    pygame.display.flip()
    clock.tick(30)
