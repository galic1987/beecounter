import numpy as np
import cv2
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Load the labels
labels = read_label_file('labels.txt')

# Load the model
interpreter = make_interpreter('model.tflite')
interpreter.allocate_tensors()

# Initialize counters
bee_in_counter = 0
bee_out_counter = 0

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Resize and reshape the frame
    _, scale = common.set_resized_input(
        interpreter, frame_pil.size, lambda size: frame_pil.resize(size, Image.ANTIALIAS))

    # Run the model
    interpreter.invoke()

    # Get the detection results
    objs = detect.get_objects(interpreter, score_threshold=0.4, image_scale=scale)

    # Count the bees
    for obj in objs:
        if labels.get(obj.id, obj.id) == 'bee':
            # TODO: Determine if the bee is entering or exiting the hive
            bee_in_counter += 1
            # bee_out_counter += 1

    # Display the frame
    cv2.imshow('frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f'Bees in: {bee_in_counter}')
print(f'Bees out: {bee_out_counter}')
