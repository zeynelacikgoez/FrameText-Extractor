import cv2
import numpy as np
import pytesseract
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple, Optional
import openai
import re
import sys
import argparse  # For command line arguments
import tiktoken
from functools import wraps
import time
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Optional: Decorator to retry on certain exceptions
def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
    """
    A decorator that retries a function upon certain exceptions.

    :param max_retries: Maximum number of retry attempts.
    :param delay: Delay between attempts in seconds.
    :param exceptions: Tuple of exceptions to handle.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    logging.warning(f"Exception {e} in {func.__name__}. Attempt {retries}/{max_retries} after {delay} seconds.")
                    time.sleep(delay)
            logging.error(f"Max number of attempts ({max_retries}) exceeded for {func.__name__}.")
            raise
        return wrapper
    return decorator

def set_tesseract_path():
    """
    Sets the path for Tesseract-OCR based on the operating system.
    Exits the program if Tesseract is not found.
    """
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
            logging.error("Tesseract-OCR is not installed or not in the PATH.")
            sys.exit(1)

def extract_text_with_pytesseract(frame: np.ndarray) -> str:
    """
    Extracts text from a given frame using Tesseract-OCR.

    :param frame: The image frame from which to extract text.
    :return: Extracted text as a string.
    """
    config = '--psm 6 --oem 1'
    try:
        text = pytesseract.image_to_string(frame, config=config)
        logging.debug(f"Extracted text: {text}")
        return text
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return ""

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, base_threshold: float = 0.05) -> bool:
    """
    Detects movement between two consecutive frames.

    :param prev_frame: The previous grayscale frame.
    :param curr_frame: The current grayscale frame.
    :param base_threshold: Base threshold for motion detection.
    :return: True if movement is detected, otherwise False.
    """
    diff = cv2.absdiff(prev_frame, curr_frame)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    average_intensity = np.mean(diff) / 255  # Normalized to [0,1]

    # Dynamic threshold based on average intensity
    dynamic_threshold = base_threshold * (1 + average_intensity)
    movement = (non_zero_count / total_pixels) > dynamic_threshold
    logging.debug(f"Movement detected: {movement} (Threshold: {dynamic_threshold:.4f}, Avg intensity: {average_intensity:.4f})")
    return movement

def preprocess_frame(frame: np.ndarray, scale_factor: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a frame by scaling, converting to grayscale, and applying a threshold.

    :param frame: The original image frame.
    :param scale_factor: Factor to reduce the size of the frame.
    :return: Tuple of the binarized frame and the grayscale frame.
    """
    try:
        resized_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_frame, gray_frame
    except cv2.error as e:
        logging.error(f"OpenCV error during preprocessing: {e}")
        return None, None

def process_frame(thresh_frame: np.ndarray) -> str:
    """
    Performs text extraction on a preprocessed frame.

    :param thresh_frame: The binarized image frame.
    :return: Extracted text as a string.
    """
    if thresh_frame is not None:
        return extract_text_with_pytesseract(thresh_frame)
    return ""

def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    """
    Estimates the number of tokens in a given text based on the used model.

    :param text: The text to be analyzed.
    :param model: The model used for tokenization.
    :return: Estimated number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.error(f"Unknown model for token estimation: {model}")
        raise
    return len(encoding.encode(text))

def process_video_optimized(
    video_path: str,
    frame_interval: float = 1.0,  # Time interval in seconds
    scale_factor: int = 2,
    motion_threshold: float = 0.05,
    max_workers: Optional[int] = None,  # New option to limit workers
    supported_formats: Optional[List[str]] = None  # Extend supported formats
) -> List[Tuple[int, str]]:
    """
    Processes a video, extracts text from relevant frames based on motion detection.

    :param video_path: Path to the video file.
    :param frame_interval: Time interval in seconds for selecting frames.
    :param scale_factor: Factor to reduce the size of the frames.
    :param motion_threshold: Threshold for motion detection.
    :param max_workers: Maximum number of workers for parallel processing.
    :param supported_formats: List of supported video formats.
    :return: List of tuples (frame number, extracted text).
    """
    if supported_formats is None:
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    extracted_text: List[Tuple[int, str]] = []

    try:
        if not os.path.exists(video_path):
            logging.error(f"Video file '{video_path}' does not exist.")
            return extracted_text

        if not os.path.isfile(video_path):
            logging.error(f"'{video_path}' is not a valid file.")
            return extracted_text

        # Check the video format
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in supported_formats:
            logging.error(f"Unsupported video format '{ext}'. Supported formats are: {supported_formats}")
            return extracted_text

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logging.error("Error opening the video file.")
                return extracted_text

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logging.error("Could not determine the video's FPS.")
                return extracted_text

            frame_interval_frames = max(1, int(fps * frame_interval))

            logging.info(f"Video FPS: {fps}")
            logging.info(f"Processing every {frame_interval_frames}th frame (equivalent to {frame_interval} seconds).")

            frame_num = 0
            frames_to_process = []
            prev_gray = None  # Initialization for motion detection

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval_frames == 0:
                    thresh_frame, gray_frame = preprocess_frame(frame, scale_factor=scale_factor)
                    if thresh_frame is not None and gray_frame is not None:
                        if prev_gray is None or detect_movement(prev_gray, gray_frame, motion_threshold):
                            frames_to_process.append((frame_num, thresh_frame))
                            logging.debug(f"Frame {frame_num} added for OCR processing.")
                        prev_gray = gray_frame

                frame_num += 1

            logging.info(f"All relevant frames ({len(frames_to_process)}) gathered for OCR processing.")

            results = []
            with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
                future_to_frame = {executor.submit(process_frame, thresh): frame_num
                                   for frame_num, thresh in frames_to_process}
                logging.info("Starting OCR processing with ThreadPoolExecutor.")

                for future in as_completed(future_to_frame):
                    frame_num = future_to_frame[future]
                    try:
                        text = future.result()
                        if text.strip():
                            results.append((frame_num, text.strip()))
                            logging.debug(f"OCR completed for frame {frame_num}.")
                    except pytesseract.TesseractError as e:
                        logging.error(f"Tesseract error at frame {frame_num}: {e}")
                    except cv2.error as e:
                        logging.error(f"OpenCV error at frame {frame_num}: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error at frame {frame_num}: {e}")

            # Sort results by frame number
            results.sort(key=lambda x: x[0])

            # Add the extracted texts
            extracted_text = results

        finally:
            cap.release()

    except Exception as e:
        logging.error(f"Error during video processing: {e}")

    return extracted_text  # Return the list of tuples (frame number, text)

def extract_output_content(text: str) -> str:
    """
    Extracts the content inside the <output> tags from the given text.

    :param text: The full text to be analyzed.
    :return: Content inside the <output> tags or the full text if no tags are found.
    """
    # Robust extraction of <output> tags
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Try to return the full text if no tags were found
        logging.warning("No <output> tags found in the LLM response. Trying to use the full text.")
        # Additional parsing logic can be added here, e.g., sentence or paragraph formation
        return text.strip()

@retry_on_exception(max_retries=3, delay=2, exceptions=(openai.error.RateLimitError, openai.error.OpenAIError,))
def correct_text_with_llm(text: str, api_key: str) -> str:
    """
    Corrects the given text using an LLM (Language Model) and returns the cleaned text.

    :param text: The text to be corrected.
    :param api_key: API key for the DeepSeek API.
    :return: Corrected text as a string.
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # **IMPORTANT: Use the correct model**
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
    except openai.error.RateLimitError as e:
        logging.error(f"RateLimitError with the DeepSeek API: {e}")
        raise  # Re-raises the exception for the decorator to handle
    except openai.error.OpenAIError as e:
        logging.error(f"DeekSeek API error: {e}")
        raise  # Re-raises the exception for the decorator to handle
    except Exception as e:
        logging.error(f"Unexpected error during LLM correction: {e}")
        raise  # Re-raises the exception

def process_and_correct_text(video_path: str, api_key: str, max_tokens: int = 7000) -> str:
    """
    Processes the video, extracts and corrects the text.

    :param video_path: Path to the video file.
    :param api_key: API key for the DeepSeek API.
    :param max_tokens: Maximum number of tokens per text chunk (including buffer).
    :return: Final corrected text.
    """
    set_tesseract_path()
    extracted_text = process_video_optimized(video_path)

    if not extracted_text:
        logging.warning("No text extracted. The corrected text will be empty.")
        return ""

    corrected_chunks = []
    current_chunk = ""
    # Ensure correct model is used with tiktoken
    try:
        encoding = tiktoken.encoding_for_model("deepseek-chat")  # Adjust for the model used
    except KeyError:
        logging.error("The model 'deepseek-chat' is not available in tiktoken.")
        sys.exit(1)

    logging.info("Starting text correction with LLM.")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame_num, text in extracted_text:
            # Token estimation including buffer
            token_count = estimate_tokens(current_chunk + " " + text) + 500
            if token_count > max_tokens:
                if current_chunk.strip():
                    futures.append(executor.submit(correct_text_with_llm, current_chunk.strip(), api_key))
                    logging.debug("Text chunk submitted for correction.")
                current_chunk = text  # Start a new chunk with the current text
            else:
                if current_chunk:
                    current_chunk += " " + text
                else:
                    current_chunk = text

        if current_chunk.strip():
            futures.append(executor.submit(correct_text_with_llm, current_chunk.strip(), api_key))
            logging.debug("Last text chunk submitted for correction.")

        for future in as_completed(futures):
            try:
                corrected_text = future.result()
                # Check if token count is within the limit
                if estimate_tokens(corrected_text) <= max_tokens:
                    corrected_chunks.append(corrected_text)
                    logging.debug("A text chunk was corrected and added.")
                else:
                    logging.warning("Corrected text exceeds the maximum token count and was not added.")
            except Exception as e:
                logging.error(f"Error correcting a text chunk: {e}")

    final_text = " ".join(corrected_chunks)
    logging.info("Text correction completed.")
    return final_text

def validate_api_key(api_key: str) -> bool:
    """
    Validates the API key by making a test call.

    :param api_key: API key to validate.
    :return: True if the key is valid, otherwise False.
    """
    if not api_key:
        logging.error("API key is not set.")
        return False
    # Optional: Make a test call to verify the key's validity
    try:
        openai.api_key = api_key
        openai.Model.list(limit=1)
        return True
    except openai.error.AuthenticationError:
        logging.error("Invalid API key for the DeepSeek API.")
    except Exception as e:
        logging.error(f"Error validating API key: {e}")
    return False

def main():
    """
    Main function to process command line arguments and start the text extraction and correction process.
    """
    parser = argparse.ArgumentParser(description="Video text extraction and correction")
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_text_path', type=str, help='Path to the output file for corrected text')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
    parser.add_argument('--frame-interval', type=float, default=1.0, help='Time interval in seconds for selecting frames')
    parser.add_argument('--scale-factor', type=int, default=2, help='Factor to reduce the size of the frames')
    parser.add_argument('--motion-threshold', type=float, default=0.05, help='Threshold for motion detection')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of workers for parallel processing')
    parser.add_argument('--supported-formats', type=str, nargs='*', default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'], help='List of supported video formats')
    parser.add_argument('--max-tokens', type=int, default=7000, help='Maximum number of tokens per text chunk (including buffer)')
    args = parser.parse_args()

    # Configure logging based on the command line argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    api_key = os.getenv("DEEPSEEK_API_KEY")  # Securely get the API key from an environment variable

    if not validate_api_key(api_key):
        logging.error("Please set the 'DEEPSEEK_API_KEY' environment variable with your actual DeepSeek API key.")
        sys.exit(1)

    try:
        final_text = process_and_correct_text(
            video_path=args.video_path,
            api_key=api_key,
            max_tokens=args.max_tokens
        )

        with open(args.output_text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        logging.info(f"Processing completed. Corrected text saved to '{args.output_text_path}'.")
    except Exception as e:
        logging.error(f"Error in the main process: {e}")

if __name__ == "__main__":
    main()
