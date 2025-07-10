import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

spi = board.SPI()
dc = digitalio.DigitalInOut(board.D6)
reset = digitalio.DigitalInOut(board.D13)
cs = digitalio.DigitalInOut(board.D5)

display = adafruit_ssd1306.SSD1306_SPI(128, 64, spi, dc, reset, cs)
display.fill(0)
display.show()

def write_to_oled(text):
    image = Image.new("1", (128, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 128, 64), fill=0)

    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (128 - w) // 2
    y = (64 - h) // 2 + 15

    draw.text((x, y), text, font=font, fill=255)
    display.image(image)
    display.show()