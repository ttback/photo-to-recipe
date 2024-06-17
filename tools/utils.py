import base64
import io
import json
from PIL import Image

def fetch_outputs(output):
    collect_streaming_outputs = []
    for o in output:
        try:
            start = o.index('{')
            jsonString = o[start:]
            d = json.loads(jsonString)
            temp = d['choices'][0]['delta']['content']
            collect_streaming_outputs.append(temp)
        except:
            pass
    outputs = ''.join(collect_streaming_outputs)
    return outputs.replace('\\', '').replace('\'', '')

def img2base64_string(img_path):
    image = Image.open(img_path)
    if image.width > 800 or image.height > 800:
        image.thumbnail((800, 800))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return image_base64

def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f"Invalid input: {s}")