import cv2
import numpy as np
import pytesseract
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple
import openai
import re
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_tesseract_path():
    if os.name == 'nt':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logging.info(f"Tesseract path set to: {tesseract_path}")
        else:
            logging.error(f"Tesseract not found at: {tesseract_path}")
            sys.exit(1)
    else:
        if not shutil.which("tesseract"):
            logging.error("Tesseract-OCR is not installed or not in PATH.")
            sys.exit(1)

def extract_text_with_pytesseract(frame: np.ndarray) -> str:
    config = '--psm 6 --oem 1'
    text = pytesseract.image_to_string(frame, config=config)
    logging.debug(f"Extracted text: {text}")
    return text

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 0.05) -> bool:
    diff = cv2.absdiff(prev_frame, curr_frame)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    movement = (non_zero_count / total_pixels) > threshold
    logging.debug(f"Movement detected: {movement} (Threshold: {threshold})")
    return movement

def preprocess_frame(frame: np.ndarray, scale_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    resized_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_frame, gray_frame

def process_frame(thresh_frame: np.ndarray) -> str:
    return extract_text_with_pytesseract(thresh_frame)

def process_video_optimized(
    video_path: str,
    frame_interval: int = 1,
    scale_factor: int = 2,
    motion_threshold: float = 0.05
) -> List[Tuple[int, str]]:
    extracted_text: List[Tuple[int, str]] = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Error opening video file.")
            return extracted_text

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_frames = int(fps) * frame_interval

        logging.info(f"Video FPS: {fps}")
        logging.info(f"Processing every {frame_interval_frames}th frame.")

        prev_gray = None
        frame_num = 0

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            logging.info("Starting video processing.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval_frames == 0:
                    thresh_frame, gray_frame = preprocess_frame(frame, scale_factor=scale_factor)
                    future = executor.submit(process_frame, thresh_frame)
                    futures[future] = (frame_num, gray_frame)
                    logging.debug(f"Frame {frame_num} submitted for OCR processing.")

                frame_num += 1

            logging.info(f"All relevant frames ({len(futures)}) submitted for OCR processing.")

            sorted_futures = sorted(futures.keys(), key=lambda f: futures[f][0])

            for future in sorted_futures:
                frame_num, current_gray = futures[future]
                try:
                    text = future.result()
                    logging.debug(f"OCR completed for frame {frame_num}.")

                    if prev_gray is not None:
                        if detect_movement(prev_gray, current_gray, motion_threshold):
                            if text.strip():
                                extracted_text.append((frame_num, text.strip()))
                                logging.info(f"Text extracted from frame {frame_num}.")
                    prev_gray = current_gray
                except pytesseract.TesseractError as e:
                    logging.error(f"Tesseract error at frame {frame_num}: {e}")
                except cv2.error as e:
                    logging.error(f"OpenCV error at frame {frame_num}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error at frame {frame_num}: {e}")

    except Exception as e:
        logging.error(f"Error processing video: {e}")
    finally:
        cap.release()
        logging.info("Video processing complete.")

    extracted_text.sort(key=lambda x: x[0])
    return [text for _, text in extracted_text]

def extract_output_content(text: str) -> str:
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def correct_text_with_llm(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant. Clean up the following text by removing all unnecessary characters "
                    "such as special symbols, double spaces, random numbers, or strings. Correct any spelling errors, "
                    "check the capitalization, and format the text for readability. Remove anything that is not relevant "
                    "to the content (e.g., timestamps or irrelevant metadata), while preserving the meaning and structure "
                    "of the original text. Wrap your output in <output> tags."
                )},
                {"role": "user", "content": text},
            ],
            stream=False
        )
        raw_response = response.choices[0].message.content
        corrected_text = extract_output_content(raw_response)
        logging.debug(f"Corrected text: {corrected_text}")
        return corrected_text
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return text

def process_and_correct_text(video_path: str, api_key: str, chunk_size: int = 1000) -> str:
    set_tesseract_path()
    extracted_text = process_video_optimized(video_path)

    openai.api_key = api_key
    openai.api_base = "https://api.deepseek.com"

    corrected_chunks = []
    current_chunk = ""

    logging.info("Starting text correction with LLM.")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for text in extracted_text:
            if len(current_chunk) + len(text) + 1 > chunk_size:
                if current_chunk.strip():
                    futures.append(executor.submit(correct_text_with_llm, current_chunk.strip()))
                    logging.debug("Text chunk submitted for correction.")
                current_chunk = text
            else:
                current_chunk += " " + text

        if current_chunk.strip():
            futures.append(executor.submit(correct_text_with_llm, current_chunk.strip()))
            logging.debug("Last text chunk submitted for correction.")

        for future in as_completed(futures):
            try:
                corrected_text = future.result()
                corrected_chunks.append(corrected_text)
                logging.debug("A text chunk has been corrected and added.")
            except Exception as e:
                logging.error(f"Error correcting a text chunk: {e}")

    final_text = " ".join(corrected_chunks)
    logging.info("Text correction complete.")
    return final_text

def main():
    video_path = "video.mp4"
    api_key = "<DeepSeek API Key>"
    output_text_path = "corrected_extracted_text.txt"

    if api_key == "<DeepSeek API Key>":
        logging.error("Please replace '<DeepSeek API Key>' with your actual OpenAI API key.")
        sys.exit(1)

    try:
        final_text = process_and_correct_text(video_path, api_key)

        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        logging.info(f"Processing complete. Corrected text saved to '{output_text_path}'.")
    except Exception as e:
        logging.error(f"Error in main process: {e}")

if __name__ == "__main__":
    main()
