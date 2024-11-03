import tkinter as tk
from tkinter import filedialog, messagebox
import json
from PIL import Image, ImageTk
import os

class QAApp:
    def __init__(self, master):
        self.master = master
        self.master.title("QA Application")

        self.data = []
        self.filtered_data = []
        self.current_index = 0
        self.answers = {}
        self.image_frames = []
        self.option_buttons = []

        # Create GUI elements
        self.create_widgets()

        # Load initial JSON data
        self.load_json()

    def create_widgets(self):
        # Frame for images
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(pady=10)

        # Frame for question text
        self.question_label = tk.Label(self.master, text="", wraplength=800, justify="left", font=("Arial", 14))
        self.question_label.pack(pady=10)

        # Frame for options
        self.input_frame = tk.Frame(self.master)
        self.input_frame.pack(pady=10)

        # Navigation buttons
        self.navigation_frame = tk.Frame(self.master)
        self.navigation_frame.pack(pady=10)

        tk.Button(self.navigation_frame, text="Previous", command=self.prev_question, font=("Arial", 12)).pack(side="left", padx=5)
        tk.Button(self.navigation_frame, text="Next", command=self.next_question, font=("Arial", 12)).pack(side="left", padx=5)
        tk.Button(self.navigation_frame, text="Load JSON", command=self.load_json, font=("Arial", 12)).pack(side="left", padx=5)
        tk.Button(self.navigation_frame, text="Load Log", command=self.load_log, font=("Arial", 12)).pack(side="left", padx=5)
        tk.Button(self.navigation_frame, text="Save Log", command=self.save_log, font=("Arial", 12)).pack(side="left", padx=5)

    def load_json(self):
        file_path = filedialog.askopenfilename(title="Select JSON File",
                                               filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            self.answers = {}
            self.filter_questions()
            self.current_index = 0
            if self.filtered_data:
                self.show_question()
            else:
                messagebox.showinfo("No Questions", "No multiple-choice or Yes/No questions found.")

    def load_log(self):
        file_path = filedialog.askopenfilename(title="Select Log File",
                                               filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            self.answers = {item['id']: item.get('answer', '') for item in self.data}
            self.filter_questions()
            # Set current_index to the first unanswered question
            for idx, item in enumerate(self.filtered_data):
                item_id = item['id']
                if not self.answers.get(item_id, ''):
                    self.current_index = idx
                    break
            else:
                # All questions are answered
                self.current_index = 0  # Start from the beginning
                messagebox.showinfo("Completed", "All questions have been answered.")
            self.show_question()

    def save_log(self):
        # Update the 'answer' key in self.data
        id_to_data = {item['id']: item for item in self.data}
        for item_id, answer in self.answers.items():
            if item_id in id_to_data:
                id_to_data[item_id]['answer'] = answer
        file_path = filedialog.asksaveasfilename(title="Save Log File",
                                                 defaultextension=".json",
                                                 filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            messagebox.showinfo("Success", "Log saved successfully.")

    def filter_questions(self):
        self.filtered_data = []
        for item in self.data:
            question = item.get('question', '').lower()
            if self.is_multiple_choice(question) or self.is_yes_no(question):
                self.filtered_data.append(item)
        if not self.filtered_data:
            self.question_label.config(text="No valid questions to display.")
            self.clear_images()
            self.clear_options()

    def is_multiple_choice(self, question):
        return any(option in question for option in ['a.', 'b.', 'c.', 'd.'])

    def is_yes_no(self, question):
        return 'yes' in question and 'no' in question

    def show_question(self):
        if not self.filtered_data:
            return
        item = self.filtered_data[self.current_index]

        # Clear previous images and options
        self.clear_images()
        self.clear_options()

        # Display all images with labels
        images = item.get('images', [])
        if images:
            self.display_images_with_labels(images)
        else:
            self.question_label.config(text='No images available.')

        # Display question
        self.question_label.config(text=item.get('question', 'No question available.'))

        # Determine question type and display options
        self.display_options(item)

    def clear_images(self):
        for frame in self.image_frames:
            frame.destroy()
        self.image_frames = []

    def clear_options(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.option_buttons = []

    def display_images_with_labels(self, images):
        # Extract camera viewpoints and images
        image_info = []
        for img_path in images:
            camera_name = self.extract_camera_name(img_path)
            image_info.append((camera_name, img_path))

        # Arrange images in order
        order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        sorted_images = sorted(image_info, key=lambda x: order.index(x[0]) if x[0] in order else len(order))

        # Create two rows
        row1 = []
        row2 = []
        for cam_name in order:
            for info in sorted_images:
                if info[0] == cam_name:
                    if cam_name in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']:
                        row1.append(info)
                    else:
                        row2.append(info)
                    break

        # Display first row
        frame_row1 = tk.Frame(self.image_frame)
        frame_row1.pack()
        self.image_frames.append(frame_row1)
        for cam_name, img_path in row1:
            self.create_image_label(frame_row1, img_path, cam_name)

        # Display second row
        frame_row2 = tk.Frame(self.image_frame)
        frame_row2.pack()
        self.image_frames.append(frame_row2)
        for cam_name, img_path in row2:
            self.create_image_label(frame_row2, img_path, cam_name)

    def create_image_label(self, parent_frame, img_path, cam_name):
        frame = tk.Frame(parent_frame)
        frame.pack(side="left", padx=5)

        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((300, 200), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack()
        else:
            label = tk.Label(frame, text='Image not found.', font=("Arial", 12))
            label.pack()

        cam_label = tk.Label(frame, text=cam_name, font=("Arial", 12))
        cam_label.pack()
        self.image_frames.append(frame)

    def extract_camera_name(self, img_path):
        # Assuming the camera name is in the file path and matches 'CAM_...'
        basename = os.path.basename(img_path)
        parts = basename.split('__')
        for part in parts:
            if part.startswith('CAM_'):
                return part
        # If not found, return 'Unknown'
        return 'Unknown'

    def display_options(self, item):
        question = item.get('question', '').lower()
        current_answer = self.answers.get(item['id'], '')

        if self.is_multiple_choice(question):
            # Multiple-choice question
            options = self.extract_options(question)
            self.option_buttons = []
            for opt_key, opt_text in options.items():
                btn = tk.Button(self.input_frame, text=f"{opt_key}", width=5, font=("Arial", 12),
                                command=lambda opt=opt_key: self.record_answer(opt))
                btn.pack(side="left", padx=5)
                if current_answer == opt_key:
                    btn.config(relief="sunken")
                else:
                    btn.config(relief="raised")
                self.option_buttons.append(btn)
        elif self.is_yes_no(question):
            # Yes/No question
            for opt in ["Yes", "No"]:
                btn = tk.Button(self.input_frame, text=opt, width=5, font=("Arial", 12),
                                command=lambda opt=opt: self.record_answer(opt))
                btn.pack(side="left", padx=5)
                if current_answer.lower() == opt.lower():
                    btn.config(relief="sunken")
                else:
                    btn.config(relief="raised")
                self.option_buttons.append(btn)

    def extract_options(self, question):
        options = {}
        lines = question.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('a.', 'b.', 'c.', 'd.', 'a)', 'b)', 'c)', 'd)')):
                opt_key = line[0].upper()
                options[opt_key] = line
        if not options:
            # Try to extract options from the same line
            if 'options:' in question:
                idx = question.find('options:')
                opts_text = question[idx + len('options:'):].strip()
                parts = opts_text.split(' ')
                for part in parts:
                    part = part.strip()
                    if part.startswith(('a.', 'b.', 'c.', 'd.', 'a)', 'b)', 'c)', 'd)')):
                        opt_key = part[0].upper()
                        options[opt_key] = part
        return options

    def record_answer(self, answer):
        item = self.filtered_data[self.current_index]
        self.answers[item['id']] = answer
        # Update buttons' relief
        for btn in self.option_buttons:
            if btn['text'].lower() == answer.lower():
                btn.config(relief="sunken")
            else:
                btn.config(relief="raised")

    def next_question(self):
        if self.current_index < len(self.filtered_data) - 1:
            self.current_index += 1
            self.show_question()
        else:
            messagebox.showinfo("End", "You have reached the last question.")

    def prev_question(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_question()
        else:
            messagebox.showinfo("Start", "You are at the first question.")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = QAApp(root)
    root.mainloop()