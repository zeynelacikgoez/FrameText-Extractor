import cv2
import numpy as np
import pytesseract
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Tuple
from openai import OpenAI
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_tesseract_path():
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_with_pillow(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    config = '--psm 6 --oem 1'
    return pytesseract.image_to_string(image, config=config)

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 0.05) -> bool:
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.mean(diff) > threshold * 255

def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame, extract_text_with_pillow(gray_frame)

def process_video_optimized(video_path: str, frame_interval: int = 1, scale_factor: int = 2, motion_threshold: float = 0.05) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error("Fehler beim Öffnen der Videodatei.")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // scale_factor)
    
    prev_frame = None
    extracted_text: List[str] = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame_num in range(0, frame_count, fps * frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (width, height))
            futures.append(executor.submit(process_frame, frame))
        
        for future in futures:
            gray_frame, text = future.result()
            
            if prev_frame is not None and detect_movement(prev_frame, gray_frame, motion_threshold):
                if text.strip():
                    extracted_text.append(text.strip())
            
            prev_frame = gray_frame
    
    cap.release()
    logging.info("Videoverarbeitung abgeschlossen.")
    return extracted_text

def extract_output_content(text: str) -> str:
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Falls keine Tags gefunden wurden, geben wir den gesamten Text zurück

def correct_text_with_llm(text: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Clean up the following text by removing all unnecessary characters such as special symbols, double spaces, random numbers, or strings. Correct any spelling errors, check the capitalization, and format the text for readability. Remove anything that is not relevant to the content (e.g., timestamps or irrelevant metadata), while preserving the meaning and structure of the original text. Wrap your output in <output> tags."},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    raw_response = response.choices[0].message.content
    return extract_output_content(raw_response)

def process_and_correct_text(video_path: str, api_key: str, chunk_size: int = 1000) -> str:
    set_tesseract_path()
    extracted_text = process_video_optimized(video_path)
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    corrected_chunks = []
    current_chunk = ""
    
    for text in extracted_text:
        if len(current_chunk) + len(text) > chunk_size:
            corrected_chunks.append(correct_text_with_llm(current_chunk, client))
            current_chunk = text
        else:
            current_chunk += " " + text
    
    if current_chunk:
        corrected_chunks.append(correct_text_with_llm(current_chunk, client))
    
    final_text = " ".join(corrected_chunks)
    return final_text

if __name__ == "__main__":
    video_path = "video.mp4"
    api_key = "<DeepSeek API Key>"
    output_text = "corrected_extracted_text.txt"
    
    final_text = process_and_correct_text(video_path, api_key)
    
    with open(output_text, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    logging.info(f"Verarbeitung abgeschlossen. Korrigierter Text in {output_text} gespeichert.")
