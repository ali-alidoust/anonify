# Anonify - Video Anonymizer

Anonify is a desktop application that automatically anonymizes people in videos by blurring their faces or full bodies. It uses AI-powered person detection with temporal masking to create smooth, consistent anonymization across video frames.

> [!CAUTION]
> You should ALWAYS double-check the resulting video for faces. AI is not perfect.

## Features

- **Smart Detection**: Uses YOLO11n AI model for accurate person detection
- **Head or Full Body**: Choose between anonymizing just heads or entire bodies
- **Temporal Masking**: Analyzes 5 frames at once for smooth, consistent blurring
- **Temporal Jitter**: Randomized blur kernels to resist AI-based deblurring
- **Audio Preservation**: Keeps original audio intact
- **Easy to Use**: Simple GUI interface with drag-and-drop support

## GUI Usage

### Getting Started

1. **Launch the Application**: Double-click the Anonify executable to start the GUI
2. The main window will appear with a simple interface

### Basic Workflow

#### 1. Select Input Video

Click the **Browse** button next to the **input** field and select the video file you want to anonymize. Supported formats include:
- MP4
- AVI
- MOV
- And other common video formats

#### 2. Choose Anonymization Options

**Detection Rate**
- Adjusts the sensitivity of person detection (0-100, default: 75)
- Higher values increase detection sensitivity, catching more people but may slow processing
- Lower values are faster but may miss people, especially distant or partially visible
- Use the slider to find the right balance for your video

**Anonymize full body instead of just head** (Optional)
- By default, Anonify blurs only the head area of detected people
- Check this box to blur entire bodies instead
- Use this for higher privacy requirements

**Disable Union Masking (over 5 frames)** (Optional)
- By default, Anonify analyzes 5 consecutive frames to create smooth, consistent masks
- Check this box to disable temporal smoothing (faster but may have flickering)
- Leave unchecked for best quality results

#### 3. Set Output Directory

Click the **Browse** button next to **output-dir** to choose where the anonymized video will be saved:
- Default location: `outputs` folder next to the executable
- The output filename will be automatically generated as `anon-XXXXXX.mp4` (where XXXXXX is a random ID)

#### 4. Process the Video

Click the **Start** button to begin processing. You'll see:
- A progress bar showing the current frame being processed
- Time remaining estimate
- The application will process each frame, applying blur to detected people

#### 5. Completion

When processing is complete:
- You'll see "Done." message
- Find your anonymized video in the output directory
- The original video remains unchanged

### Tips

- **Processing Time**: Videos take time to process (approximately 1-5 seconds per frame depending on your computer)
- **Quality Settings**: Keep temporal masking enabled for best results
- **File Size**: Output videos are saved in H.264 format with good compression
- **Privacy**: Head-only mode is sufficient for most privacy needs; full-body mode is more aggressive

### Troubleshooting

**Video processing fails**
- Ensure the input video file is not corrupted
- Check that you have write permissions to the output directory
- Make sure you have enough disk space for the output file

**Blur quality issues**
- Enable temporal masking for smoother results
- Use full-body mode if faces are not detected reliably

## System Requirements

- Windows 10 or later
- Sufficient RAM for video processing (4GB+ recommended)
- Storage space for output videos

## Privacy Note

Anonify processes videos locally on your computer. No data is sent to external servers. The AI model (YOLO11n) runs entirely offline.

## Attribution

Icon source: [Flaticon](https://www.flaticon.com/)