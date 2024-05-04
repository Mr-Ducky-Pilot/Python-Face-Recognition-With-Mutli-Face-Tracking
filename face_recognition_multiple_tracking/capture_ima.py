import cv2
import os
import pyaudio
import wave
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
import threading
import dlib
import time

class FaceRecognition:
    def __init__(self):
        self.BASE_DIR = "loved_ones_data"
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.name = ""
        self.recognized_persons = []
        self.last_played_time = {}

        self.root = tk.Tk()
        self.root.title("Your Loved Ones")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.home_page()

    def home_page(self):
        self.clear_widgets()

        add_person_button = tk.Button(self.root, text="Add A New Person", command=self.add_person_page)
        add_person_button.pack(pady=20)

        recognize_button = tk.Button(self.root, text="Face Recognition", command=self.recognize_face)
        recognize_button.pack()

    def add_person_page(self):
        self.clear_widgets()

        name_label = tk.Label(self.root, text="Enter name of loved one:")
        name_label.pack()

        name_entry = tk.Entry(self.root)
        name_entry.pack()

        def capture_data():
            self.name = name_entry.get()
            self.capture_image()

        capture_button = tk.Button(self.root, text="Begin Face Recognition", command=capture_data)
        capture_button.pack(pady=20)

    def capture_image(self):
        folder_path = os.path.join(self.BASE_DIR, self.name)
        os.makedirs(folder_path, exist_ok=True)

        cam = cv2.VideoCapture(0)

        for i in range(50):
            _, image = cam.read()
            cv2.imwrite(os.path.join(folder_path, f"{self.name}_{i}.jpg"), image)
            cv2.imshow(f"Capturing Face: {self.name}", image)
            cv2.waitKey(100)

        cam.release()
        cv2.destroyAllWindows()

        messagebox.showinfo("Capture Audio", "Press OK to start capturing audio.")
        self.record_audio(folder_path)

    def record_audio(self, folder_path):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.SAMPLE_RATE, input=True,
                            frames_per_buffer=self.CHUNK_SIZE)

        frames = []

        for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK_SIZE * 3)):
            data = stream.read(self.CHUNK_SIZE)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(os.path.join(folder_path, f"{self.name}.wav"), 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.SAMPLE_RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        messagebox.showinfo("Success", "Face and audio data captured successfully.")
        self.train_model()
        self.home_page()

    def train_model(self):
        known_encodings = []
        known_names = []

        for person_name in os.listdir(self.BASE_DIR):
            person_dir = os.path.join(self.BASE_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            for image_name in os.listdir(person_dir):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(person_dir, image_name)
                    image = dlib.load_rgb_image(image_path)
                    faces = self.face_detector(image, 1)
                    if len(faces) > 0:
                        shape = self.shape_predictor(image, faces[0])
                        face_encoding = self.face_recognizer.compute_face_descriptor(image, shape)
                        known_encodings.append(np.array(face_encoding))
                        known_names.append(person_name)

        self.known_encodings = known_encodings
        self.known_names = known_names

        np.save(os.path.join(self.BASE_DIR, 'known_encodings.npy'), known_encodings)
        with open(os.path.join(self.BASE_DIR, 'known_names.pkl'), 'wb') as file:
            pickle.dump(known_names, file)


    def recognize_face(self):
        if not hasattr(self, 'known_encodings') or not hasattr(self, 'known_names'):
            encodings_path = os.path.join(self.BASE_DIR, 'known_encodings.npy')
            names_path = os.path.join(self.BASE_DIR, 'known_names.pkl')
            if os.path.exists(encodings_path) and os.path.exists(names_path):
                self.known_encodings = np.load(encodings_path)
                with open(names_path, 'rb') as file:
                    self.known_names = pickle.load(file)
            else:
                messagebox.showinfo("No Trained Model", "No trained model found. Please add a person first.")
                return

        self.clear_widgets()

        self.recognized_persons = []
        self.cam = cv2.VideoCapture(0)

        def face_recognition_thread():
            while True:
                _, frame = self.cam.read()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_detector(rgb_frame, 1)

                current_recognized_persons = []

                for face in faces:
                    shape = self.shape_predictor(rgb_frame, face)
                    face_encoding = self.face_recognizer.compute_face_descriptor(rgb_frame, shape)
                    face_encoding = np.array(face_encoding)
                    distances = np.linalg.norm(self.known_encodings - face_encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                    name = "Unknown"

                    if distances[min_distance_index] < 0.6:
                        name = self.known_names[min_distance_index]

                    left = face.left()
                    top = face.top()
                    right = face.right()
                    bottom = face.bottom()

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    current_recognized_persons.append(name)

                    if name not in self.recognized_persons and name != "Unknown":
                        self.recognized_persons.append(name)
                        current_time = time.time()
                        last_played = self.last_played_time.get(name, 0)
                        if current_time - last_played > 60:  # Check if audio was played more than 1 min ago
                            audio_path = os.path.join(self.BASE_DIR, name, f"{name}.wav")
                            if os.path.exists(audio_path):
                                self.play_audio(audio_path)
                                self.last_played_time[name] = current_time

                for person_name in self.recognized_persons:
                    if person_name not in current_recognized_persons:
                        self.recognized_persons.remove(person_name)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)

                self.camera_label.config(image=img)
                self.camera_label.image = img

        self.camera_label = tk.Label(self.root)
        self.camera_label.pack()

        stop_button = tk.Button(self.root, text="Stop", command=self.stop_recognition)
        stop_button.pack(pady=10)

        face_recognition_thread = threading.Thread(target=face_recognition_thread)
        face_recognition_thread.daemon = True
        face_recognition_thread.start()


    def stop_recognition(self):
        self.running = False
        self.cam.release()
        cv2.destroyAllWindows()
        self.home_page()


    def play_audio(self, audio_path):
        def audio_thread():
            audio = pyaudio.PyAudio()
            wave_file = wave.open(audio_path, 'rb')
            stream = audio.open(format=audio.get_format_from_width(wave_file.getsampwidth()),
                                channels=wave_file.getnchannels(),
                                rate=wave_file.getframerate(),
                                output=True)

            data = wave_file.readframes(self.CHUNK_SIZE)
            while data:
                stream.write(data)
                data = wave_file.readframes(self.CHUNK_SIZE)

            stream.stop_stream()
            stream.close()
            audio.terminate()

        threading.Thread(target=audio_thread).start()

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    face_recognition = FaceRecognition()
    face_recognition.run()



# ## Smart Glasses Dlib. Works. Small video lag cause of frame rate
# import cv2
# import os
# import pyaudio
# import wave
# import numpy as np
# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import pickle
# import threading
# import dlib

# class FaceRecognition:
#     def __init__(self):
#         self.BASE_DIR = "loved_ones_data"
#         self.CHUNK_SIZE = 1024
#         self.FORMAT = pyaudio.paInt16
#         self.CHANNELS = 1
#         self.SAMPLE_RATE = 44100
#         self.name = ""
#         self.recognized_persons = []

#         self.root = tk.Tk()
#         self.root.title("Your Loved Ones")
#         self.root.geometry("800x600")
#         self.root.resizable(False, False)

#         self.face_detector = dlib.get_frontal_face_detector()
#         self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#         self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#         self.home_page()

#     def home_page(self):
#         self.clear_widgets()

#         add_person_button = tk.Button(self.root, text="Add A New Person", command=self.add_person_page)
#         add_person_button.pack(pady=20)

#         recognize_button = tk.Button(self.root, text="Face Recognition", command=self.recognize_face)
#         recognize_button.pack()

#     def add_person_page(self):
#         self.clear_widgets()

#         name_label = tk.Label(self.root, text="Enter name of loved one:")
#         name_label.pack()

#         name_entry = tk.Entry(self.root)
#         name_entry.pack()

#         def capture_data():
#             self.name = name_entry.get()
#             self.capture_image()

#         capture_button = tk.Button(self.root, text="Begin Face Recognition", command=capture_data)
#         capture_button.pack(pady=20)

#     def capture_image(self):
#         folder_path = os.path.join(self.BASE_DIR, self.name)
#         os.makedirs(folder_path, exist_ok=True)

#         cam = cv2.VideoCapture(0)

#         for i in range(50):
#             _, image = cam.read()
#             cv2.imwrite(os.path.join(folder_path, f"{self.name}_{i}.jpg"), image)
#             cv2.imshow(f"Capturing Face: {self.name}", image)
#             cv2.waitKey(100)

#         cam.release()
#         cv2.destroyAllWindows()

#         messagebox.showinfo("Capture Audio", "Press OK to start capturing audio.")
#         self.record_audio(folder_path)

#     def record_audio(self, folder_path):
#         audio = pyaudio.PyAudio()
#         stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
#                             rate=self.SAMPLE_RATE, input=True,
#                             frames_per_buffer=self.CHUNK_SIZE)

#         frames = []

#         for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK_SIZE * 3)):
#             data = stream.read(self.CHUNK_SIZE)
#             frames.append(data)

#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

#         wave_file = wave.open(os.path.join(folder_path, f"{self.name}.wav"), 'wb')
#         wave_file.setnchannels(self.CHANNELS)
#         wave_file.setsampwidth(audio.get_sample_size(self.FORMAT))
#         wave_file.setframerate(self.SAMPLE_RATE)
#         wave_file.writeframes(b''.join(frames))
#         wave_file.close()

#         messagebox.showinfo("Success", "Face and audio data captured successfully.")
#         self.train_model()
#         self.home_page()

#     def train_model(self):
#         known_encodings = []
#         known_names = []

#         for person_name in os.listdir(self.BASE_DIR):
#             person_dir = os.path.join(self.BASE_DIR, person_name)
#             if not os.path.isdir(person_dir):
#                 continue
#             for image_name in os.listdir(person_dir):
#                 if image_name.endswith(".jpg"):
#                     image_path = os.path.join(person_dir, image_name)
#                     image = dlib.load_rgb_image(image_path)
#                     faces = self.face_detector(image, 1)
#                     if len(faces) > 0:
#                         shape = self.shape_predictor(image, faces[0])
#                         face_encoding = self.face_recognizer.compute_face_descriptor(image, shape)
#                         known_encodings.append(np.array(face_encoding))
#                         known_names.append(person_name)

#         self.known_encodings = known_encodings
#         self.known_names = known_names

#         np.save(os.path.join(self.BASE_DIR, 'known_encodings.npy'), known_encodings)
#         with open(os.path.join(self.BASE_DIR, 'known_names.pkl'), 'wb') as file:
#             pickle.dump(known_names, file)


#     def recognize_face(self):
#         if not hasattr(self, 'known_encodings') or not hasattr(self, 'known_names'):
#             encodings_path = os.path.join(self.BASE_DIR, 'known_encodings.npy')
#             names_path = os.path.join(self.BASE_DIR, 'known_names.pkl')
#             if os.path.exists(encodings_path) and os.path.exists(names_path):
#                 self.known_encodings = np.load(encodings_path)
#                 with open(names_path, 'rb') as file:
#                     self.known_names = pickle.load(file)
#             else:
#                 messagebox.showinfo("No Trained Model", "No trained model found. Please add a person first.")
#                 return

#         self.clear_widgets()

#         self.recognized_persons = []
#         self.cam = cv2.VideoCapture(0)
#         self.frame_count = 0

#         def update_frame():
#             _, frame = self.cam.read()
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces = self.face_detector(rgb_frame, 1)

#             current_recognized_persons = []

#             for face in faces:
#                 shape = self.shape_predictor(rgb_frame, face)
#                 face_encoding = self.face_recognizer.compute_face_descriptor(rgb_frame, shape)
#                 face_encoding = np.array(face_encoding)
#                 distances = np.linalg.norm(self.known_encodings - face_encoding, axis=1)
#                 min_distance_index = np.argmin(distances)
#                 name = "Unknown"

#                 if distances[min_distance_index] < 0.6:
#                     name = self.known_names[min_distance_index]

#                 left = face.left()
#                 top = face.top()
#                 right = face.right()
#                 bottom = face.bottom()

#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#                 cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                 current_recognized_persons.append(name)

#                 if name not in self.recognized_persons and name != "Unknown":
#                     self.recognized_persons.append(name)
#                     audio_path = os.path.join(self.BASE_DIR, name, f"{name}.wav")
#                     if os.path.exists(audio_path):
#                         self.play_audio(audio_path)

#             for person_name in self.recognized_persons:
#                 if person_name not in current_recognized_persons:
#                     self.recognized_persons.remove(person_name)

#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             img = ImageTk.PhotoImage(img)

#             self.camera_label.config(image=img)
#             self.camera_label.image = img

#             self.update_frame_id = self.camera_label.after(10, update_frame)

#         self.camera_label = tk.Label(self.root)
#         self.camera_label.pack()

#         stop_button = tk.Button(self.root, text="Stop", command=self.stop_recognition)
#         stop_button.pack(pady=10)

#         self.update_frame_id = self.camera_label.after(10, update_frame)


#     def stop_recognition(self):
#         self.camera_label.after_cancel(self.update_frame_id)
#         self.cam.release()
#         cv2.destroyAllWindows()
#         self.home_page()

#     def play_audio(self, audio_path):
#         def audio_thread():
#             audio = pyaudio.PyAudio()
#             wave_file = wave.open(audio_path, 'rb')
#             stream = audio.open(format=audio.get_format_from_width(wave_file.getsampwidth()),
#                                 channels=wave_file.getnchannels(),
#                                 rate=wave_file.getframerate(),
#                                 output=True)

#             data = wave_file.readframes(self.CHUNK_SIZE)
#             while data:
#                 stream.write(data)
#                 data = wave_file.readframes(self.CHUNK_SIZE)

#             stream.stop_stream()
#             stream.close()
#             audio.terminate()

#         threading.Thread(target=audio_thread).start()

#     def clear_widgets(self):
#         for widget in self.root.winfo_children():
#             widget.destroy()

#     def run(self):
#         self.root.mainloop()

# if __name__ == "__main__":
#     face_recognition = FaceRecognition()
#     face_recognition.run()







# # Works Perfectly. Fixed Auido playback causing video to lag by creating a seperate thread

# import cv2
# import os
# import pyaudio
# import wave
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# # pickle for labelling the trained face data
# import pickle
# # threading to create a seperate thread process to play the audio to avoid video stream lagging.
# import threading


# class FaceRecognition:
#     def __init__(self):
#         self.BASE_DIR = "loved_ones_data"  # Create a folder for storing data
#         self.CHUNK_SIZE = 1024
#         self.FORMAT = pyaudio.paInt16
#         self.CHANNELS = 1
#         self.SAMPLE_RATE = 44100
#         self.name = ""
#         self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#         self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#         self.label_encoder = LabelEncoder()

#         # Keep track of recognised face
#         self.recognized_persons = []

#         self.root = tk.Tk()
#         self.root.title("Your Loved Ones")
#         self.root.geometry("800x600")
#         self.root.resizable(False, False)

#         self.home_page()

#     def home_page(self):
#         self.clear_widgets()

#         add_person_button = tk.Button(self.root, text="Add A New Person", command=self.add_person_page)
#         add_person_button.pack(pady=20)

#         recognize_button = tk.Button(self.root, text="Face Recognition", command=self.recognize_face)
#         recognize_button.pack()

#     def add_person_page(self):
#         self.clear_widgets()

#         name_label = tk.Label(self.root, text="Enter name of loved one:")
#         name_label.pack()

#         name_entry = tk.Entry(self.root)
#         name_entry.pack()

#         def capture_data():
#             self.name = name_entry.get()
#             self.capture_image()

#         capture_button = tk.Button(self.root, text="Begin Face Recognition", command=capture_data)
#         capture_button.pack(pady=20)

#     def capture_image(self):
#         folder_path = os.path.join(self.BASE_DIR, self.name)
#         os.makedirs(folder_path, exist_ok=True)

#         cam = cv2.VideoCapture(0)  # Use primary camera

#         for i in range(300):
#             _, image = cam.read()
#             cv2.imwrite(os.path.join(folder_path, f"{self.name}_{i}.jpg"), image)
#             cv2.imshow(f"Capturing Face: {self.name}", image)
#             cv2.waitKey(100)

#         cam.release()
#         cv2.destroyAllWindows()

#         messagebox.showinfo("Capture Audio", "Press OK to start capturing audio.")
#         self.record_audio(folder_path)

#     def record_audio(self, folder_path):
#         audio = pyaudio.PyAudio()
#         stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
#                             rate=self.SAMPLE_RATE, input=True,
#                             frames_per_buffer=self.CHUNK_SIZE)

#         frames = []

#         for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK_SIZE * 3)):  # Record for 3 seconds
#             data = stream.read(self.CHUNK_SIZE)
#             frames.append(data)

#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

#         wave_file = wave.open(os.path.join(folder_path, f"{self.name}.wav"), 'wb')
#         wave_file.setnchannels(self.CHANNELS)
#         wave_file.setsampwidth(audio.get_sample_size(self.FORMAT))
#         wave_file.setframerate(self.SAMPLE_RATE)
#         wave_file.writeframes(b''.join(frames))
#         wave_file.close()

#         messagebox.showinfo("Success", "Face and audio data captured successfully.")
#         self.train_model()
#         self.home_page()

#     def train_model(self):
#         faces = []
#         labels = []

#         for person_name in os.listdir(self.BASE_DIR):
#             person_dir = os.path.join(self.BASE_DIR, person_name)
#             if not os.path.isdir(person_dir):
#                 continue  # Skip files that are not directories
#             for image_name in os.listdir(person_dir):
#                 if image_name.endswith(".jpg"):
#                     image_path = os.path.join(person_dir, image_name)
#                     image = cv2.imread(image_path)
#                     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                     face = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#                     if len(face) > 0:
#                         (x, y, w, h) = face[0]
#                         face_roi = gray[y:y+h, x:x+w]
#                         faces.append(face_roi)
#                         labels.append(person_name)

#         encoded_labels = self.label_encoder.fit_transform(labels)
#         self.face_recognizer.train(faces, np.array(encoded_labels))
#          # Save the trained model to a file
#         model_path = os.path.join(self.BASE_DIR, 'trained_model.yml')
#         self.face_recognizer.write(model_path)

#         # Save the fitted LabelEncoder to a file
#         encoder_path = os.path.join(self.BASE_DIR, 'label_encoder.pkl')
#         with open(encoder_path, 'wb') as file:
#             pickle.dump(self.label_encoder, file)

#     def recognize_face(self):
#         if not hasattr(self.face_recognizer, 'labels_'):
#             # If the model is not trained, try to load the trained model from a file
#             model_path = os.path.join(self.BASE_DIR, 'trained_model.yml')
#             if os.path.exists(model_path):
#                 self.face_recognizer.read(model_path)
                
#                 # Load the fitted LabelEncoder from a file
#                 encoder_path = os.path.join(self.BASE_DIR, 'label_encoder.pkl')
#                 if os.path.exists(encoder_path):
#                     with open(encoder_path, 'rb') as file:
#                         self.label_encoder = pickle.load(file)
#                 else:
#                     messagebox.showinfo("No Fitted LabelEncoder", "No fitted LabelEncoder found. Please add a person first.")
#                     return
#             else:
#                 messagebox.showinfo("No Trained Model", "No trained model found. Please add a person first.")
#                 return

#         self.clear_widgets()

#         self.recognized_persons = []  # Change: Use a list to keep track of recognized persons
#         self.cam = cv2.VideoCapture(0)  # Change: Store the camera object in self.cam

#         def update_frame():
#             _, frame = self.cam.read()  # Change: Use self.cam instead of cam
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             current_recognized_persons = []  # Change: Keep track of currently recognized persons

#             for (x, y, w, h) in faces:
#                 face_roi = gray[y:y+h, x:x+w]
#                 label, confidence = self.face_recognizer.predict(face_roi)
#                 person_name = self.label_encoder.inverse_transform([label])[0]

#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                 current_recognized_persons.append(person_name)  # Change: Add recognized person to current list

#                 if person_name not in self.recognized_persons:  # Change: Check if person is not already recognized
#                     self.recognized_persons.append(person_name)  # Change: Add person to recognized list
#                     audio_path = os.path.join(self.BASE_DIR, person_name, f"{person_name}.wav")
#                     if os.path.exists(audio_path):
#                         self.play_audio(audio_path)

#             # Change: Remove persons who are no longer in the frame
#             for person_name in self.recognized_persons:
#                 if person_name not in current_recognized_persons:
#                     self.recognized_persons.remove(person_name)

#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             img = ImageTk.PhotoImage(img)

#             self.camera_label.config(image=img)
#             self.camera_label.image = img

#             self.update_frame_id = self.camera_label.after(10, update_frame)  # Change: Store the after ID

#         self.camera_label = tk.Label(self.root)
#         self.camera_label.pack()

#         stop_button = tk.Button(self.root, text="Stop", command=self.stop_recognition)
#         stop_button.pack(pady=10)

#         self.update_frame_id = self.camera_label.after(10, update_frame)  # Change: Store the initial after ID


#     def stop_recognition(self):
#         self.camera_label.after_cancel(self.update_frame_id)
#         self.cam.release()
#         cv2.destroyAllWindows()
#         self.home_page()

#     ##Old Audio code which caused the video frame to lag
#     # def play_audio(self, audio_path):
#     #     audio = pyaudio.PyAudio()
#     #     wave_file = wave.open(audio_path, 'rb')
#     #     stream = audio.open(format=audio.get_format_from_width(wave_file.getsampwidth()),
#     #                         channels=wave_file.getnchannels(),
#     #                         rate=wave_file.getframerate(),
#     #                         output=True)

#     #     data = wave_file.readframes(self.CHUNK_SIZE)
#     #     while data:
#     #         stream.write(data)
#     #         data = wave_file.readframes(self.CHUNK_SIZE)

#     #     stream.stop_stream()
#     #     stream.close()
#     #     audio.terminate()

#     ## New Audio code with seperate thread to avoid vidoe lag
#     def play_audio(self, audio_path):
#         def audio_thread():
#             audio = pyaudio.PyAudio()
#             wave_file = wave.open(audio_path, 'rb')
#             stream = audio.open(format=audio.get_format_from_width(wave_file.getsampwidth()),
#                                 channels=wave_file.getnchannels(),
#                                 rate=wave_file.getframerate(),
#                                 output=True)

#             data = wave_file.readframes(self.CHUNK_SIZE)
#             while data:
#                 stream.write(data)
#                 data = wave_file.readframes(self.CHUNK_SIZE)

#             stream.stop_stream()
#             stream.close()
#             audio.terminate()

#         threading.Thread(target=audio_thread).start()

#     def clear_widgets(self):
#         for widget in self.root.winfo_children():
#             widget.destroy()

#     def run(self):
#         self.root.mainloop()

# # Usage
# if __name__ == "__main__":
#     face_recognition = FaceRecognition()
#     face_recognition.run()
