# FrameText Extractor

FrameText Extractor is an open-source tool for optimized text extraction (OCR) from videos. It combines OpenCV, Pillow, and Tesseract to extract text from individual video frames, using multithreading to improve processing performance.

## Features

- **Text Recognition (OCR)**: Extracts text from video frames using Tesseract OCR.
- **Optimized Video Processing**: Processes frames at regular intervals (e.g., 1 frame per second) and uses multithreading for better performance.
- **Motion Detection**: Detects changes between frames to avoid unnecessary text extraction on static frames.
- **Scalable Processing**: Utilizes all available CPU cores for faster execution.
- **Flexible Customization**: Allows for dynamic adjustment of frame interval, frame size, and motion detection sensitivity.

## Requirements

To use this project, you'll need:

- Python 3.x
- [OpenCV](https://opencv.org/) (`cv2`)
- [Pillow](https://python-pillow.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (installed and available in the system path)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [Numpy](https://numpy.org/)

If you are using Windows, ensure that Tesseract is installed and the path is set correctly.

### Installing Dependencies

You can install the necessary Python libraries with:

```bash
pip install opencv-python pillow pytesseract numpy
```

Install Tesseract OCR:

- **Windows**: [Tesseract Download](https://github.com/tesseract-ocr/tesseract/wiki)
- **macOS**: Install via Homebrew:
  ```bash
  brew install tesseract
  ```
- **Linux**: Install via your system’s package manager (e.g., `apt` on Ubuntu):
  ```bash
  sudo apt install tesseract-ocr
  ```

## Usage

1. **Set Tesseract Path (Windows only)**:
   Update the path in the `set_tesseract_path()` function if necessary:

   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

2. **Process Video**:
   Place your video in the same directory or specify the path in the `video_path` variable.

3. **Run the Script**:

   ```bash
   python frametext_extractor.py
   ```

   The extracted text will be saved to the output file specified in `output_text`.

## Example Code

```python
if __name__ == "__main__":
    set_tesseract_path()
    video_path = "video.mp4"
    output_text = "optimized_extracted_text.txt"
    process_video_optimized(video_path, output_text, frame_interval=1, scale_factor=2, motion_threshold=0.05)
```

## How it Works

1. **Load Video**: The video is loaded, and frames are processed at regular intervals (e.g., 1 frame per second).
2. **Resize Frames**: Frames are resized to speed up processing.
3. **Motion Detection**: The script checks if the current frame differs significantly from the previous frame to avoid unnecessary OCR operations.
4. **Text Extraction**: If motion is detected, the text is extracted using Tesseract OCR.
5. **Save Results**: The extracted text is saved to a text file.

## Customization

You can customize the following parameters to suit your needs:

- **Frame Interval**: Process more or fewer frames by adjusting the interval between frames. This is done by setting the `frame_interval` parameter when calling the `process_video_optimized` function:
  
  ```python
  process_video_optimized(video_path, output_text, frame_interval=2)
  ```
  This example processes one frame every two seconds (if `fps = 1`).

- **Frame Size**: Adjust the scaling of the frames to influence processing time. Use the `scale_factor` parameter to resize frames. For example, to resize frames to one-third of their original size:
  
  ```python
  process_video_optimized(video_path, output_text, scale_factor=3)
  ```

- **Motion Threshold**: Adjust the sensitivity of motion detection by changing the `motion_threshold` parameter. A higher threshold reduces sensitivity (i.e., fewer movements are detected), while a lower threshold increases sensitivity:
  
  ```python
  process_video_optimized(video_path, output_text, motion_threshold=0.1)
  ```

### Full Example:
You can combine all these options to fine-tune your video processing:

```python
process_video_optimized(video_path, output_text, frame_interval=1, scale_factor=2, motion_threshold=0.05)
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you have improvements or find bugs.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
