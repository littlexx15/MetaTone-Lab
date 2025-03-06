import io
import tempfile
from PIL import Image

def create_temp_file(pil_image: Image.Image) -> str:
    """
    将一个 PIL Image 对象保存到临时 PNG 文件，并返回文件路径。
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pil_image.save(tmp, format="PNG")
        tmp_path = tmp.name
    return tmp_path

def get_image_bytes(image_file: str) -> bytes:
    """
    打开指定路径的图片文件，并返回其字节数据。
    """
    image = Image.open(image_file)
    with io.BytesIO() as output:
        image.save(output, format="PNG")  # 可根据需求改成 "JPEG" 等
        image_bytes = output.getvalue()
    return image_bytes
