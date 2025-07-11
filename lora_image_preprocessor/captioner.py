# lora_image_preprocessor/captioner.py
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

processor = None
model = None

def load_caption_model():
    global processor, model
    if processor is None or model is None:
        print("Loading captioning model...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )
        model.eval()
        print("Captioning model loaded.")

def generate_caption(pil_image):
    global processor, model
    if processor is None or model is None:
        load_caption_model()

    prompt = "USER: <image>\nDescribe this image in terms of appearance, emotions, age, ethnicity, hair and clothing of the character. Do not describe the background.\nASSISTANT:"
    
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Clean up the output
    assistant_response_start = "ASSISTANT:"
    caption_start_index = caption.find(assistant_response_start)
    if caption_start_index != -1:
        caption = caption[caption_start_index + len(assistant_response_start):].strip()
        
    print("Done")
    return caption
