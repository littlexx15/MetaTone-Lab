import io
import tempfile
from PIL import Image

def create_temp_file(pil_image: Image.Image) -> str:
    """
    Saves a PIL Image object to a temporary PNG file and returns the file path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pil_image.save(tmp, format="PNG")
        tmp_path = tmp.name
    return tmp_path

def get_image_bytes(image_file: str) -> bytes:
    """
    Opens the image file at the specified path and returns its byte data.
    """
    image = Image.open(image_file)
    with io.BytesIO() as output:
        # Change format to "JPEG" or others as needed
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    return image_bytes
