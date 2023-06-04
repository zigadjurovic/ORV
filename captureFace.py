import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 as the argument for the default camera

# Define the output directory to save the captured frames
output_dir = 'captured_frames/viktor/viktor'

# Initialize frame count and capture count
frame_count = 0
capture_count = 0

ret, frame = cap.read()
cv2.imshow('Frame', frame)
cv2.imwrite(output_dir + 'frame_{:04d}.jpg'.format(capture_count), frame)

# Release the camera capture object and close any open windows
cap.release()
cv2.destroyAllWindows()