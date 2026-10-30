import cv2


def detect_and_measure_all_shapes(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at path '{image_path}'.")
        return

    output = img.copy()

    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use bilateral filter or blur to reduce noise while keeping edges crisp
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's Thresholding automatically calculates the optimal threshold value
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Optional: Morphological operations to separate closely touching shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # 3. Find Contours
    # RETR_EXTERNAL gets only top-level outer contours (prevents double-counting holes)
    contours, _ = cv2.findContours(
        thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    shape_count = 0

    for contour in contours:
        # Ignore small noisy dots or the entire image border
        area = cv2.contourArea(contour)
        if area < 300:  # Adjust minimum area threshold if shapes are tiny
            continue

        perimeter = cv2.arcLength(contour, True)

        # Approximate shape vertices
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)

        # Classify shape
        shape_name = "Unknown"
        if num_vertices == 3:
            shape_name = "Triangle"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif num_vertices == 5:
            shape_name = "Pentagon"
        elif num_vertices == 6:
            shape_name = "Hexagon"
        else:
            shape_name = "Circle"

        shape_count += 1

        # Draw outline
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

        # Compute centroid for placing text labels
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = approx[0][0]

        # Render labels inside shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 0, 0)
        thickness = 1

        cv2.putText(
            output,
            f"{shape_name}",
            (cX - 25, cY - 12),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.putText(
            output,
            f"A:{int(area)}",
            (cX - 25, cY + 2),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.putText(
            output,
            f"P:{int(perimeter)}",
            (cX - 25, cY + 16),
            font,
            font_scale,
            color,
            thickness,
        )

    print(f"Total shapes detected: {shape_count}")

    # 4. Display result
    cv2.imshow("Multiple Shapes Detected", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function call
detect_and_measure_all_shapes("shapes.png")
