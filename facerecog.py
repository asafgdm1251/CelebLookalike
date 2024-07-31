import gradio as gr 
from deepface import DeepFace 
import glob 
import os 
from PIL import Image

CELEB_IMAGES_DIR = './celebs'

def find_similar_face(user_image):
    
    #find all images in subdirectories 
    celeb_images = glob.glob(os.path.join(CELEB_IMAGES_DIR, "*", "*.jpg"))
    
    best_match = None
    best_distance = float("inf")
    
    for celeb_image in celeb_images:
        try: 
            print(f"Processing {celeb_image}...")
            result = DeepFace.verify(user_image, celeb_image, model_name="VGG-Face")
            #lower the distance means the faces are more similar 
            distance = result['distance']
            print(f"Distance for {celeb_image}: {distance}")
            
            if(distance < best_distance):
                best_distance = distance 
                best_match = celeb_image
        
        except Exception as e: 
            print(f"Error processiong {celeb_image}: {e}")
            
    if best_match:
        print(f"Best match: {best_match} with distance {best_distance}")
        return Image.open(best_match)
        
iface = gr.Interface(
    fn = find_similar_face,
    inputs = gr.Image(type="filepath"),
    outputs=gr.Image(),
    title="Find Your Celebrity Look-Alike",
    description="Upload your photo to see which celebrity you resemble the most."
)

iface.launch()