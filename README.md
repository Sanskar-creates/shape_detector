# shape_detector
detects shapes from the image  
# OpenCV Shape Detection with Area & Perimeter Calculation

A Python-based computer vision tool using OpenCV (`cv2`) to detect, classify, and measure geometric shapes from an input image. The script automatically identifies multiple shapes, calculates their area and perimeter, and displays the measurements directly inside each detected shape in real time.

---

## Features

- **Multi-Shape Classification:** Detects triangles, squares, rectangles, pentagons, hexagons, and circles.
- **Dynamic Thresholding:** Uses Otsu's Thresholding to automatically adjust to different image lighting conditions and contrast.
- **Metric Measurements:** Calculates the **Area** (in square pixels) and **Perimeter** (in pixels) for every detected contour.
- **Contour Noise Filtering:** Ignores minor image noise and small artifacts using customizable area thresholds.
- **Visual Overlay:** Draws vector outlines around detected shapes and displays labeled text (Shape Name, Area, Perimeter) centered at each shape's centroid.

---

## Detected Shapes

| Shape | Vertices Count | Classification Logic |
| :--- | :--- | :--- |
| **Triangle** | 3 | Polygon approximation |
| **Square** | 4 | Aspect ratio ($W/H$) between $0.95$ and $1.05$ |
| **Rectangle** | 4 | Aspect ratio ($W/H$) $< 0.95$ or $> 1.05$ |
| **Pentagon** | 5 | Polygon approximation |
| **Hexagon** | 6 | Polygon approximation |
| **Circle** | $> 6$ | Curves with high vertex count |

---

## Prerequisites

Ensure you have Python installed on your system. Install the required dependencies using `pip`:

```bash
pip install opencv-python numpy
