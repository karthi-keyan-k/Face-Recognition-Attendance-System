import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Load the known faces and their names from your dataset
dataset_dir = r'C:\Users\sange\Documents\dataset_dir'
known_face_encodings = []
known_face_names = []


print ("\n Welcome to Face Recognition Attendance System")
print ("\n ---------------------------------------------")



for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)

        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Find the face encodings
        face_encoding = face_recognition.face_encodings(image)

        if len(face_encoding) > 0:
            # Add the first face encoding found (assuming one face per image)
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(filename)  # Use the filename as the known face name



# Create a CSV file for attendance
csv_file_path = r"C:\Users\sange\Documents\attend\attendance.csv"

# Set to keep track of identified faces
identified_faces = set()

# Function to save attendance record
def save_attendance(filename):
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the CSV file in append mode
    with open(csv_file_path, mode="a", newline="") as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the attendance record if not already identified
        if filename not in identified_faces:
            csv_writer.writerow([filename, current_datetime])
            identified_faces.add(filename)  # Add to the set of identified faces

        print (filename)
# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the webcam
    ret, frame = video_capture.read()

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Compare this face with the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        filename = "Unknown"  # Default filename if no match is found

        # If a match is found, use the filename from the known faces
        if True in matches:
            first_match_index = matches.index(True)
            filename = known_face_names[first_match_index]

            # Save attendance record
            save_attendance(filename)

        # Draw a rectangle around the face and label with the filename
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, filename, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
