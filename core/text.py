from PIL import Image, ImageDraw, ImageFont
import freetype as ftp
import numpy as np

def text2Image(text, img_size, font_size, font):
    im = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, font_size)
    _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
    position = ((img_size - text_width) // 2, (img_size - text_height) // 2)
    draw.text(position, text, font=font, fill=255)
    return im


def char2Polygon(char, font, font_size=13, scale=64):
    face = ftp.Face(font)
    flags = ftp.FT_LOAD_DEFAULT | ftp.FT_LOAD_NO_BITMAP
    face.set_char_size(font_size*scale)
    face.load_char(char, flags)
    slot = face.glyph
    outline = slot.outline
    contours = [0]
    contours.extend(outline.contours)
    contours_arr = []
    points = outline.points
    for i in range(len(contours)-1):
        if i == 0:
            arr = points[contours[i]:contours[i+1]+1]
            arr.append(arr[0])
        else:
            arr = points[contours[i]+1:contours[i+1]+1]
            arr.append(arr[0])
            arr = arr[::-1]
        contours_arr.append(np.array(arr)/scale)
    return contours_arr


def str2Polygons(string: str, font, font_size=13, scale=128, _return_offsets=False):
    polygons = []
    total_offset_x = 0
    total_offset_y = 0
    for char in string:
        if char == "\n":
            total_offset_y -= 3*font_size/2
            total_offset_x = 0
        else:
            if char != " ":
                char_polygons = char2Polygon(char, font, font_size, scale)
                translated_polygons = [polygon + np.array([total_offset_x, total_offset_y])
                                       for polygon in char_polygons]
                polygons.extend(translated_polygons)
            total_offset_x += font_size
    if _return_offsets:
        return polygons, total_offset_x+font_size, total_offset_y-3*font_size/2
    return polygons
