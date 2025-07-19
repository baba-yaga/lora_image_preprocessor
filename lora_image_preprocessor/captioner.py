import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BlipForConditionalGeneration, BlipProcessor
from PIL import Image

# LLaVA Model
LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
llava_processor = None
llava_model = None

# BLIP Model
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
blip_processor = None
blip_model = None


def load_llava_model():
    global llava_processor, llava_model
    if llava_processor is None or llava_model is None:
        print("Loading LLaVA captioning model...")
        llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_NAME)
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )
        llava_model.eval()
        print("LLaVA captioning model loaded.")

def load_blip_model():
    global blip_processor, blip_model
    if blip_processor is None or blip_model is None:
        print("Loading BLIP captioning model...")
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            torch_dtype=torch.float16
        ).to("cuda")
        blip_model.eval()
        print("BLIP captioning model loaded.")


def generate_llava_caption(pil_image):
    global llava_processor, llava_model
    if llava_processor is None or llava_model is None:
        load_llava_model()

    prompt = "USER: <image>\nDescribe this image in details: the look, emotions, age, etnicity, skin color, hair and clothing of the character, the image style, like for AI image generator how to recreate it, be very accurate. Do not describe the background. Do not include any introduction.\nASSISTANT:"
    
    inputs = llava_processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        output = llava_model.generate(**inputs, max_new_tokens=200)
    
    caption = llava_processor.decode(output[0], skip_special_tokens=True)
    
    # Clean up the output
    assistant_response_start = "ASSISTANT:"
    caption_start_index = caption.find(assistant_response_start)
    if caption_start_index != -1:
        caption = caption[caption_start_index + len(assistant_response_start):].strip()
        
    print("LLaVA caption generated.")
    return caption

def generate_blip_caption(pil_image):
    global blip_processor, blip_model
    if blip_processor is None or blip_model is None:
        load_blip_model()
    
    # Generate a caption with a generic prompt
    prompt = "a photograph of"
    inputs = blip_processor(pil_image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=50)

    # Decode the output and clean it up
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    
    # The prompt is often included in the output, so we remove it.
    caption = caption.replace(prompt, "").strip()

    print("BLIP caption generated.")
    return caption

def generate_captions(pil_image):
    blip_caption = generate_blip_caption(pil_image)
    llava_caption = generate_llava_caption(pil_image)
    return f"{blip_caption}\n\n{llava_caption}"
