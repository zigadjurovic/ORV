import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 as the argument for the default camera

# Define the output directory to save the captured frames
output_dir = 'ORV/captured_frames/viktor/viktor'

# Initialize frame count and capture count
frame_count = 0
capture_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Frame', frame)

    # Capture every 10th frame (adjust this as per your requirement)
    if frame_count % 10 == 0:
        # Save the frame as an image file
        cv2.imwrite(output_dir + 'frame_{:04d}.jpg'.format(capture_count), frame)
        capture_count += 1

    frame_count += 1

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
