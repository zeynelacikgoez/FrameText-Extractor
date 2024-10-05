import cv2
import numpy as np
import pytesseract
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from typing import List, Tuple, Optional
import openai
import re
import sys
import argparse  # Für die Kommandozeilenargumente
import tiktoken

# Setze Logging-Ebene auf DEBUG für detaillierte Logs (standardmäßig, kann über args geändert werden)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_tesseract_path():
    if os.name == 'nt':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logging.info(f"Tesseract-Pfad gesetzt zu: {tesseract_path}")
        else:
            logging.error(f"Tesseract nicht gefunden unter: {tesseract_path}")
            sys.exit(1)
    else:
        if not shutil.which("tesseract"):
            logging.error("Tesseract-OCR ist nicht installiert oder nicht im PATH.")
            sys.exit(1)

def extract_text_with_pytesseract(frame: np.ndarray) -> str:
    config = '--psm 6 --oem 1'
    try:
        text = pytesseract.image_to_string(frame, config=config)
        logging.debug(f"Extrahierter Text: {text}")
        return text
    except (pytesseract.TesseractError, pytesseract.pytesseract.TesseractNotFoundError, pytesseract.pytesseract.SubprocessError) as e:
        logging.error(f"Tesseract-Fehler während OCR: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unerwarteter Fehler bei Tesseract-OCR: {e}")
        return ""

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, base_threshold: float = 0.05) -> bool:
    diff = cv2.absdiff(prev_frame, curr_frame)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    average_intensity = np.mean(diff) / 255  # Normalisiert auf [0,1]

    # Dynamischer Schwellenwert basierend auf durchschnittlicher Intensität
    dynamic_threshold = base_threshold * (1 + average_intensity)
    movement = (non_zero_count / total_pixels) > dynamic_threshold
    logging.debug(f"Bewegung erkannt: {movement} (Schwelle: {dynamic_threshold:.4f}, Durchschnittliche Intensität: {average_intensity:.4f})")
    return movement

def preprocess_frame(frame: np.ndarray, scale_factor: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        resized_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_frame, gray_frame
    except cv2.error as e:
        logging.error(f"OpenCV-Fehler während der Vorverarbeitung: {e}")
        return None, None

def process_frame(thresh_frame: np.ndarray) -> str:
    if thresh_frame is not None:
        return extract_text_with_pytesseract(thresh_frame)
    return ""

def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def process_video_optimized(
    video_path: str,
    frame_interval: float = 1.0,  # Zeitintervall in Sekunden
    scale_factor: int = 2,
    motion_threshold: float = 0.05,
    max_workers: Optional[int] = None  # Neue Option zur Begrenzung der Worker
) -> List[Tuple[int, str]]:
    extracted_text: List[Tuple[int, str]] = []

    try:
        if not os.path.exists(video_path):
            logging.error(f"Video-Datei '{video_path}' existiert nicht.")
            return extracted_text

        if not os.path.isfile(video_path):
            logging.error(f"'{video_path}' ist keine gültige Datei.")
            return extracted_text

        # Überprüfe das Video-Format
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in supported_formats:
            logging.error(f"Unsupported video format '{ext}'. Supported formats are: {supported_formats}")
            return extracted_text

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logging.error("Fehler beim Öffnen der Videodatei.")
                return extracted_text

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logging.error("FPS des Videos konnte nicht ermittelt werden.")
                return extracted_text

            frame_interval_frames = max(1, int(fps * frame_interval))

            logging.info(f"Video FPS: {fps}")
            logging.info(f"Verarbeite jedes {frame_interval_frames}. Frame (entspricht {frame_interval} Sekunden).")

            prev_gray = None
            frame_num = 0

            with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
                futures = {}
                logging.info("Starte Videoverarbeitung.")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_num % frame_interval_frames == 0:
                        thresh_frame, gray_frame = preprocess_frame(frame, scale_factor=scale_factor)
                        if thresh_frame is not None and gray_frame is not None:
                            future = executor.submit(process_frame, thresh_frame)
                            futures[future] = (frame_num, gray_frame)
                            logging.debug(f"Frame {frame_num} zur OCR-Verarbeitung eingereicht.")

                    frame_num += 1

                logging.info(f"Alle relevanten Frames ({len(futures)}) zur OCR-Verarbeitung eingereicht.")

                for future in as_completed(futures):
                    frame_num, current_gray = futures[future]
                    try:
                        text = future.result()
                        logging.debug(f"OCR abgeschlossen für Frame {frame_num}.")

                        if prev_gray is not None:
                            if detect_movement(prev_gray, current_gray, motion_threshold):
                                if text.strip():
                                    extracted_text.append((frame_num, text.strip()))
                                    logging.info(f"Text aus Frame {frame_num} extrahiert.")
                        prev_gray = current_gray
                    except pytesseract.TesseractError as e:
                        logging.error(f"Tesseract-Fehler bei Frame {frame_num}: {e}")
                    except cv2.error as e:
                        logging.error(f"OpenCV-Fehler bei Frame {frame_num}: {e}")
                    except Exception as e:
                        logging.error(f"Unerwarteter Fehler bei Frame {frame_num}: {e}")

        finally:
            cap.release()

    except Exception as e:
        logging.error(f"Fehler bei der Videoverarbeitung: {e}")

    extracted_text.sort(key=lambda x: x[0])
    return extracted_text  # Rückgabe der Liste von Tupeln (Frame-Nummer, Text)

def extract_output_content(text: str) -> str:
    # Robuste Extraktion der <output>-Tags
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Versuche, das gesamte Text zurückzugeben, falls keine Tags gefunden wurden
        logging.warning("Keine <output>-Tags in der LLM-Antwort gefunden. Versuche, den gesamten Text zu verwenden.")
        # Optional: Weitere Parsing-Logik hier hinzufügen
        return text.strip()

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
        logging.debug(f"Korrigierter Text: {corrected_text}")
        return corrected_text
    except openai.error.OpenAIError as e:
        logging.error(f"DeepSeek API-Fehler: {e}")
        return text
    except Exception as e:
        logging.error(f"Unerwarteter Fehler bei der Korrektur mit LLM: {e}")
        return text

def process_and_correct_text(video_path: str, api_key: str, max_tokens: int = 7500) -> str:
    set_tesseract_path()
    extracted_text = process_video_optimized(video_path)

    openai.api_key = api_key

    corrected_chunks = []
    current_chunk = ""
    encoding = tiktoken.encoding_for_model("deepseek-chat")

    logging.info("Starte Textkorrektur mit LLM.")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame_num, text in extracted_text:
            if estimate_tokens(current_chunk + " " + text) > max_tokens:
                if current_chunk.strip():
                    futures.append(executor.submit(correct_text_with_llm, current_chunk.strip()))
                    logging.debug("Text-Chunk zur Korrektur eingereicht.")
                current_chunk = text
            else:
                current_chunk += " " + text

        if current_chunk.strip():
            futures.append(executor.submit(correct_text_with_llm, current_chunk.strip()))
            logging.debug("Letzter Text-Chunk zur Korrektur eingereicht.")

        for future in as_completed(futures):
            try:
                corrected_text = future.result()
                corrected_chunks.append(corrected_text)
                logging.debug("Ein Text-Chunk wurde korrigiert und hinzugefügt.")
            except Exception as e:
                logging.error(f"Fehler bei der Korrektur eines Text-Chunks: {e}")

    final_text = " ".join(corrected_chunks)
    logging.info("Textkorrektur abgeschlossen.")
    return final_text

def main():
    parser = argparse.ArgumentParser(description="Video-Text-Extraktion und Korrektur")
    parser.add_argument('video_path', type=str, help='Pfad zur Videodatei')
    parser.add_argument('output_text_path', type=str, help='Pfad zur Ausgabedatei für den korrigierten Text')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Setzen der Logging-Ebene')
    args = parser.parse_args()

    # Konfiguriere Logging basierend auf dem Kommandozeilenargument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Ungültige Logging-Ebene: {args.log_level}")
    logging.getLogger().setLevel(numeric_level)

    api_key = os.getenv("DEEPSEEK_API_KEY")  # API-Schlüssel sicher über Umgebungsvariable

    if not api_key or api_key == "<DeepSeek API Key>":
        logging.error("Bitte setzen Sie die Umgebungsvariable 'DEEPSEEK_API_KEY' mit Ihrem tatsächlichen DeepSeek API-Schlüssel.")
        sys.exit(1)

    try:
        final_text = process_and_correct_text(args.video_path, api_key)

        with open(args.output_text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        logging.info(f"Verarbeitung abgeschlossen. Korrigierter Text gespeichert unter '{args.output_text_path}'.")
    except Exception as e:
        logging.error(f"Fehler im Hauptprozess: {e}")

if __name__ == "__main__":
    main()
