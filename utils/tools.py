import io
import cv2
import base64

from PIL import Image
from io import BytesIO

def img_to_base64(self, img):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def base64_to_img(msg):
    msg = base64.b64decode(msg)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img