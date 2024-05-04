import cv2
import os
import pyaudio
import wave
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
import threading
import face_recognition


class FaceRecognition:
    def __init__(self):
        self.BASE_DIR = "loved_ones_data"
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SAMPLE_RATE = 44100
        self.name = ""
        self.label_encoder = LabelEncoder()
        self.recognized_persons = []

        self.root = tk.Tk()
        self.root.title("Your Loved Ones")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

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
        known_face_encodings = []
        known_face_names = []

        for person_name in os.listdir(self.BASE_DIR):
            person_dir = os.path.join(self.BASE_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            for image_name in os.listdir(person_dir):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(person_dir, image_name)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)

        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

        model_path = os.path.join(self.BASE_DIR, 'trained_model.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump((known_face_encodings, known_face_names), file)

    def recognize_face(self):
        if not hasattr(self, 'known_face_encodings') or not hasattr(self, 'known_face_names'):
            model_path = os.path.join(self.BASE_DIR, 'trained_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    self.known_face_encodings, self.known_face_names = pickle.load(file)
            else:
                messagebox.showinfo("No Trained Model", "No trained model found. Please add a person first.")
                return

        self.clear_widgets()
        self.recognized_persons = []
        self.cam = cv2.VideoCapture(0)

        def update_frame():
            _, frame = self.cam.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if name not in self.recognized_persons:
                    self.recognized_persons.append(name)
                    audio_path = os.path.join(self.BASE_DIR, name, f"{name}.wav")
                    if os.path.exists(audio_path):
                        self.play_audio(audio_path)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.camera_label.config(image=img)
            self.camera_label.image = img

            self.update_frame_id = self.camera_label.after(10, update_frame)

        self.camera_label = tk.Label(self.root)
        self.camera_label.pack()

        stop_button = tk.Button(self.root, text="Stop", command=self.stop_recognition)
        stop_button.pack(pady=10)

        self.update_frame_id = self.camera_label.after(10, update_frame)

    def stop_recognition(self):
        self.camera_label.after_cancel(self.update_frame_id)
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

# Usage
if __name__ == "__main__":
    face_recognition = FaceRecognition()
    face_recognition.run()

