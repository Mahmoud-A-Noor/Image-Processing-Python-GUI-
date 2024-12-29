# PyImageLab

A comprehensive image processing application that provides various image manipulation techniques and filters. This desktop application enables users to apply different image processing operations, view histograms, and save processed images with multiple theme options.

---

## Project Preview
![Simple Project Overview](https://user-images.githubusercontent.com/59361888/210313373-861c87db-2716-4bdb-82b8-33c9df21a00a.png)
---
## Key Features

- **Image Processing Operations**: Multiple image manipulation techniques including grayscale conversion, thresholding, and contrast adjustment
- **Advanced Filters**: Various filters including Average, Median, Sobel, and Gaussian filters
- **Histogram Analysis**: Real-time histogram visualization for both original and processed images
- **Multiple Themes**: 12+ customizable themes for the user interface
- **Accumulative Effects**: Support for applying multiple techniques with undo capability
- **Save Processed Images**: Export capabilities for processed images

---

## Technologies Used

### GUI
- **PyQt5**: Main GUI framework for building the desktop application
- **Qt Designer**: Used for designing the user interface (GUI.ui file)
- **Matplotlib**: For plotting image histograms and visualizations

### Image Processing
- **OpenCV (cv2)**: Core image processing library
- **NumPy**: Numerical computing library for image manipulation
- **Python**: Primary programming language (3.9+)

### Development Tools
- **pyqt-tools**: Additional tools for PyQt development
- **QSS**: Qt Style Sheets for theming

---

## Project Structure
```
PyImageLab/
├── GUI.ui              # Qt Designer UI file
├── main.py            # Main application code
├── themes/            # Theme style sheets
│   ├── AMOLED.qss
│   ├── Aqua.qss
│   ├── ...
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

---

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/PyImageLab.git
cd PyImageLab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. From the File menu, browse for an input image
2. Select desired image processing techniques from the left sidebar
3. Apply multiple techniques (effects are cumulative)
4. Use the arrow icon to revert changes if needed
5. Save the processed image using the File menu when done

**Note**: Change the application theme from the Theme menu (12+ themes available)

## Available Techniques

### Basic Operations
- RGB to Grayscale Conversion
- Thresholding
- Resampling (Up/Down)
- Gray Level Adjustment
- Negative Transform
- Log Transform
- Power Law Transform
- Contrast Enhancement
- Gray Level Slicing
- Arithmetic Operations (Add/Subtract)
- Logical Operations
- Bit Plane Slicing
- Histogram Equalization

### Filters
- Average Filter
- Min/Max Filter
- Median Filter
- Weighted Average Filter
- First Derivative Filter
- Composite Laplacian Filter
- Sobel Operator Filter
- Ideal Low/High Pass Filter
- Butterworth Low/High Pass Filter
- Gaussian Low/High Pass Filter

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or feedback, please reach out to mahmoudnoor917@gmail.com.
