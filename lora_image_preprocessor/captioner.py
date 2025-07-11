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

def generate_caption(pil_image, blendshapes=None):
    global processor, model
    if processor is None or model is None:
        load_caption_model()

    prompt = "USER: <image>\nDescribe this image in terms of appearance, emotions, ethnicity, hair and clothing of the character"
    
    if blendshapes:
        emotion_prompt = " The person's expression is a mix of"
        for emotion, score in blendshapes.items():
            if score > 0.3: # Threshold to only include significant emotions
                emotion_prompt += f" {emotion} ({score:.2f}),"
        prompt += emotion_prompt[:-1] + "." # remove last comma

    prompt += "\nASSISTANT:"
    
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
