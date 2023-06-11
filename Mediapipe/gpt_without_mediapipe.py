import cv2
import numpy as np

# Define the region of interest (ROI) for hand tracking
top, right, bottom, left = 0, 0, 400, 590

# Initialize the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    # Extract the ROI for hand tracking
    roi = frame[top:bottom, right:left]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Perform background subtraction
    _, threshold = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are detected
    if contours:
        # Find the contour with the maximum area
        max_contour = max(contours, key=cv2.contourArea)

        # Find the convex hull of the contour
        hull = cv2.convexHull(max_contour)

        # Draw the contour and convex hull on the ROI
        cv2.drawContours(roi, [max_contour], 0, (0, 255, 0), 2)
        cv2.drawContours(roi, [hull], 0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
