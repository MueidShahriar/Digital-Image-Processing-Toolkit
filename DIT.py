import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import threading
from numpy.lib.stride_tricks import sliding_window_view
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # For PyInstaller
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# CORE IMAGE PROCESSING ALGORITHMS (Manual Implementations) 
def convert_to_grayscale(img_array):
    if img_array.ndim == 3 and img_array.shape[2] >= 3:
        grayscale_array = (
            0.299 * img_array[:, :, 0] +
            0.587 * img_array[:, :, 1] +
            0.114 * img_array[:, :, 2]
        ).astype(np.uint8)
        return grayscale_array
    return img_array.astype(np.uint8)

def apply_convolution(image_array, kernel):
    if image_array.ndim == 3:
        image_array = convert_to_grayscale(image_array)

    kernel = np.asarray(kernel, dtype=np.float32)
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    windows = sliding_window_view(padded_image, (kH, kW))
    convolved = np.tensordot(windows, kernel[::-1, ::-1], axes=((2, 3), (0, 1)))

    return np.clip(convolved, 0, 255).astype(np.uint8)

# SPECIFIC OPERATOR FUNCTIONS
def image_negative(img_array):
    return (255 - img_array).astype(np.uint8)

def smoothing_filter(img_array, kernel_size=3):
    if kernel_size % 2 == 0: kernel_size += 1 
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return apply_convolution(img_array, kernel)

def sharpening_filter(img_array):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return apply_convolution(img_array, kernel)

def edge_detection_filter(img_array):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    return apply_convolution(img_array, kernel)

def apply_thresholding(img_array, T=128):
    if img_array.ndim == 3:
        img_array = convert_to_grayscale(img_array)
        
    output_array = np.zeros_like(img_array)
    output_array[img_array > T] = 255
    output_array[img_array <= T] = 0
    
    return output_array.astype(np.uint8)

def apply_log_transformation(img_array, c=25):
    if img_array.ndim == 3:
        img_array = convert_to_grayscale(img_array)

    img_float = img_array.astype(np.float32)
    normalized_array = img_float / 255.0
    
    log_transformed = c * np.log(1 + normalized_array)
    
    # Scale back to 0-255
    max_val = np.max(log_transformed)
    if max_val == 0:
        return np.zeros_like(img_array, dtype=np.uint8)
        
    output_array = (log_transformed / max_val * 255).astype(np.uint8)
    return output_array

def apply_gamma_transformation(img_array, gamma=0.5):
    if img_array.ndim == 3:
        img_array = convert_to_grayscale(img_array)

    img_float = img_array.astype(np.float32)
    normalized_array = img_float / 255.0
    
    # Apply S = R^gamma
    gamma_transformed = np.power(normalized_array, gamma)
    
    # Scale back to 0-255
    output_array = (gamma_transformed * 255).astype(np.uint8)
    return output_array

def calculate_histogram(img_array):
    if img_array.ndim == 3:
        img_array = convert_to_grayscale(img_array)
        
    counts, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # Plot the histogram using matplotlib
    plt.figure(figsize=(6, 4))
    plt.title("Image Histogram", fontsize=14)
    plt.xlabel("Pixel Intensity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.bar(bins[:-1], counts, width=1.0, color='#6D5A4B', alpha=0.8)
    plt.xlim([0, 255])
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    
    return img_array 

def apply_resizing(img, new_width, new_height):
    try:
        new_width = int(new_width)
        new_height = int(new_height)
        if new_width > 0 and new_height > 0:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except ValueError:
        print("Invalid resize parameters.")
    return img

# TKINTER GUI APPLICATION 
class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Digital Image Processing Toolkit")
        master.geometry("1000x800")
        
        self.original_img = None
        self.processed_img = None
        self.current_filepath = ""
        self.is_processing = False

        # State Variables for Parameters
        self.smoothing_kernel_size = tk.StringVar(value="3")
        self.threshold_value = tk.StringVar(value="128")
        self.log_c_value = tk.StringVar(value="25")
        self.gamma_value = tk.StringVar(value="0.5")
        self.resize_width = tk.StringVar(value="256")
        self.resize_height = tk.StringVar(value="256")
        
        self.panel_max_width = 420
        self.panel_max_height = 420

        # Define developer info variables inside the class
        self.DEVELOPER_NAME = "MD. MUEID SHAHRIAR"
        self.DEVELOPER_ID = "0812220205101016"
        self.DEVELOPER_PHOTO_PATH = resource_path("mueid.jpg")
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.master, padx=15, pady=15, bg="#FDFBF7")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_rowconfigure(2, weight=0)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)

        # Developer Info Panel 
        dev_frame = tk.Frame(main_frame, bd=1, relief=tk.RIDGE, bg="#114C67")
        dev_frame.grid(row=0, column=0, rowspan=3, padx=(0, 15), pady=5, sticky="nsw")
        dev_frame.grid_rowconfigure(0, weight=1)
        dev_frame.grid_columnconfigure(0, weight=1)

        tk.Label(dev_frame, text="Developer", font=('Inter', 11, 'bold'), bg="#191F20", fg="#FFFFFF", pady=4).pack(fill="x")

        card_body = tk.Frame(dev_frame, bg="#114C67", bd=0)
        card_body.pack(padx=10, pady=8, fill="both", expand=True)

        try:
            # Load developer photo 
            dev_photo_pil = Image.open(self.DEVELOPER_PHOTO_PATH).resize((70, 90), Image.Resampling.LANCZOS)
            self.dev_photo_tk = ImageTk.PhotoImage(dev_photo_pil)
            photo_label = tk.Label(card_body, image=self.dev_photo_tk, bd=0, bg="#F3E6D6")
            photo_label.pack(pady=(0, 6))
        except FileNotFoundError:
            photo_label = tk.Label(card_body, text="[Your Photo]", width=12, height=6, bg="#D8C8B8", fg="#6D5A4B", bd=1, relief=tk.GROOVE)
            photo_label.pack(pady=(0, 6))
        except Exception as e:
            print(f"Error loading photo: {e}")
            photo_label = tk.Label(card_body, text="[Photo Error]", width=12, height=6, bg="#D8C8B8", fg="#6D5A4B", bd=1, relief=tk.GROOVE)
            photo_label.pack(pady=(0, 6))

        tk.Label(card_body, text=self.DEVELOPER_NAME, font=('Inter', 10, 'bold'), bg="#114C67", fg="#FFFFFF").pack()
        tk.Label(card_body, text=self.DEVELOPER_ID, font=('Inter', 9), bg="#114C67", fg="#FFFFFF").pack(pady=(2, 0))
        ttk.Separator(card_body, orient="horizontal").pack(fill="x", pady=8)

        # Content Column 
        right_frame = tk.Frame(main_frame, bg="#FDFBF7")
        right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=0)
        right_frame.grid_rowconfigure(2, weight=0)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Image Display Frame 
        image_frame = tk.Frame(right_frame, bg="#FDFBF7")
        image_frame.grid(row=0, column=0, pady=10, sticky="nsew")
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        
        tk.Label(image_frame, text="Original Image", font=('Inter', 12, 'bold'), bg="#FDFBF7").grid(row=0, column=0, padx=5, pady=5)
        self.original_container = tk.Frame(image_frame, width=self.panel_max_width, height=self.panel_max_height, bg='#EAEAEA', relief=tk.SUNKEN, bd=1)
        self.original_container.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.original_container.grid_propagate(False)
        self.original_panel = tk.Label(self.original_container, text="Load an image...", bg='#EAEAEA', anchor='center')
        self.original_panel.pack(expand=True, fill='both')
        
        tk.Label(image_frame, text="Processed Image", font=('Inter', 12, 'bold'), bg="#FDFBF7").grid(row=0, column=1, padx=5, pady=5)
        self.processed_container = tk.Frame(image_frame, width=self.panel_max_width, height=self.panel_max_height, bg='#EAEAEA', relief=tk.SUNKEN, bd=1)
        self.processed_container.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.processed_container.grid_propagate(False)
        self.processed_panel = tk.Label(self.processed_container, text="Run an operation...", bg='#EAEAEA', anchor='center')
        self.processed_panel.pack(expand=True, fill='both')

        status_frame = tk.Frame(right_frame, bg="#FDFBF7")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status_frame, textvariable=self.status_var, font=('Inter', 9), bg="#FDFBF7", fg="#444").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, padx=5, pady=3, sticky='e')
        self.progress.grid_remove()

        # Control Panel
        control_frame = tk.Frame(right_frame, bd=2, relief=tk.FLAT, bg="#FFFFFF", padx=10, pady=10)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Function to create buttons with consistent styling
        def create_button(text, command, color):
            return tk.Button(control_frame, text=text, command=command, bg=color, fg="white", 
                            font=('Inter', 10, 'bold'), relief=tk.FLAT, padx=10, pady=5, 
                            activebackground=color, activeforeground="white")
        
        # Button layout setup
        btn_row = 0
        create_button("Upload Image", self.load_image, "#277429").grid(row=btn_row, column=0, padx=5, pady=5, sticky='ew')
        create_button("Save Image", self.save_image, "#174B76").grid(row=btn_row, column=1, padx=5, pady=5, sticky='ew')
        create_button("Reset Image", self.reset_image, "#8B2A23").grid(row=btn_row, column=2, padx=5, pady=5, sticky='ew')
        create_button("Negative", lambda: self.run_operation(image_negative, "Negative"), "#D94316").grid(row=btn_row, column=3, padx=5, pady=5, sticky='ew')
        create_button("Sharpening", lambda: self.run_operation(sharpening_filter, "Sharpening"), "#645013").grid(row=btn_row, column=4, padx=5, pady=5, sticky='ew')
        create_button("Edge Detect", lambda: self.run_operation(edge_detection_filter, "Edge Detection"), "#561761").grid(row=btn_row, column=5, padx=5, pady=5, sticky='ew')
        create_button("Histogram", lambda: self.run_operation(calculate_histogram, "Histogram"), "#283C47").grid(row=btn_row, column=6, padx=5, pady=5, sticky='ew')
        
        # Parameterized Operations (Row 1)
        param_row = 1
        
        # Smoothing Controls
        tk.Label(control_frame, text="Smooth (Kernel Size):", bg="#FFFFFF", fg="#4A4A4A").grid(row=param_row, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(control_frame, textvariable=self.smoothing_kernel_size, width=5).grid(row=param_row, column=1, padx=5, sticky='w')
        create_button("Apply Smooth", lambda: self.run_operation(smoothing_filter, "Smoothing", kernel_size=int(self.smoothing_kernel_size.get())), "#183235").grid(row=param_row, column=2, padx=5, pady=5, sticky='ew')
        
        # Thresholding Controls
        tk.Label(control_frame, text="Threshold (T):", bg="#FFFFFF", fg="#4A4A4A").grid(row=param_row, column=3, padx=5, pady=5, sticky='w')
        tk.Entry(control_frame, textvariable=self.threshold_value, width=5).grid(row=param_row, column=4, padx=5, sticky='w')
        create_button("Apply Threshold", lambda: self.run_operation(apply_thresholding, "Thresholding", T=int(self.threshold_value.get())), "#293C14").grid(row=param_row, column=5, padx=5, pady=5, sticky='ew')

        # Parameterized Operations (Row 2)
        param_row = 2
        
        # Log Transform Controls
        tk.Label(control_frame, text="Log (C Factor):", bg="#FFFFFF", fg="#4A4A4A").grid(row=param_row, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(control_frame, textvariable=self.log_c_value, width=5).grid(row=param_row, column=1, padx=5, sticky='w')
        create_button("Apply Log", lambda: self.run_operation(apply_log_transformation, "Log Transform", c=float(self.log_c_value.get())), "#483027").grid(row=param_row, column=2, padx=5, pady=5, sticky='ew')
        
        # Gamma Transform Controls
        tk.Label(control_frame, text="Gamma (γ):", bg="#FFFFFF", fg="#4A4A4A").grid(row=param_row, column=3, padx=5, pady=5, sticky='w')
        tk.Entry(control_frame, textvariable=self.gamma_value, width=5).grid(row=param_row, column=4, padx=5, sticky='w')
        create_button("Apply Gamma", lambda: self.run_operation(apply_gamma_transformation, "Gamma Transform", gamma=float(self.gamma_value.get())), "#56132A").grid(row=param_row, column=5, padx=5, pady=5, sticky='ew')
        
        # Resizing Controls
        param_row = 3
        tk.Label(control_frame, text="Resize (W x H):", bg="#FFFFFF", fg="#4A4A4A").grid(row=param_row, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(control_frame, textvariable=self.resize_width, width=5).grid(row=param_row, column=1, padx=2, sticky='w')
        tk.Entry(control_frame, textvariable=self.resize_height, width=5).grid(row=param_row, column=1, padx=(30, 0), sticky='w')
        create_button("Apply Resize", self.run_resize, "#331764").grid(row=param_row, column=2, padx=5, pady=5, sticky='ew')

    def clear_processed_panel(self):
        self.processed_panel.config(image="", text="Run an operation...")
        self.processed_panel.image = None
        self.processed_img = None

    def load_image(self):
        f_types = [('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]
        filepath = filedialog.askopenfilename(filetypes=f_types)
        if filepath:
            try:
                self.current_filepath = filepath
                self.original_img = Image.open(filepath).convert('RGB')
                self.processed_img = None
                self.update_display(self.original_img, self.original_panel, is_original=True)
                self.clear_processed_panel()
            except Exception as e:
                print(f"Error loading image: {e}")
                
    def reset_image(self):
        if self.original_img is None:
            print("No image loaded to reset.")
            return
        
        self.processed_img = self.original_img.copy()
        self.update_display(self.processed_img, self.processed_panel, is_original=False)
        print("Processed image reset to original.")

    def save_image(self):
        if not self.processed_img:
            print("No image processed yet to save.")
            return

        default_name = self.current_filepath.rsplit('.', 1)[0] + "_processed.png" if self.current_filepath else "processed_image.png"
            
        f_types = [('PNG File', '*.png'), ('JPEG File', '*.jpg')]
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=f_types, initialfile=default_name)
        
        if filepath:
            try:
                self.processed_img.save(filepath)
                print(f"Image saved to {filepath}")
            except Exception as e:
                print(f"Error saving image: {e}")

    def update_display(self, img, panel, is_original):
        max_width = self.panel_max_width
        max_height = self.panel_max_height

        w, h = img.size
        scale = min(max_width / w, max_height / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        
        resized_img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized_img)

        panel.config(image=tk_img, text="")
        panel.image = tk_img

    def show_loader(self, message):
        self.status_var.set(message)
        self.progress.grid()
        self.progress.start(10)
        self.master.config(cursor="watch")
        self.master.update_idletasks()
        self.is_processing = True

    def hide_loader(self):
        self.progress.stop()
        self.progress.grid_remove()
        self.master.config(cursor="")
        self.is_processing = False

    def _complete_operation(self, image, operation_name):
        self.processed_img = image
        self.update_display(self.processed_img, self.processed_panel, is_original=False)
        self.status_var.set(f"{operation_name} completed.")

    def _handle_error(self, message):
        print(message)
        self.status_var.set(message)

    def _start_background_task(self, worker, message):
        if self.is_processing:
            print("Another operation is already running. Please wait.")
            return False
        self.show_loader(message)
        threading.Thread(target=worker, daemon=True).start()
        return True

    def _run_operation_worker(self, operation_func, operation_name, kwargs):
        try:
            source_img = self.processed_img if self.processed_img is not None else self.original_img
            img_array = np.array(source_img)
            result = operation_func(img_array, **kwargs)
            if isinstance(result, Image.Image):
                output_image = result
            elif result.ndim == 3 and result.shape[2] == 3:
                output_image = Image.fromarray(result.astype(np.uint8), 'RGB')
            else:
                output_image = Image.fromarray(result.astype(np.uint8)).convert('L')
            self.master.after(0, lambda: self._complete_operation(output_image, operation_name))
        except ValueError as err:
            self.master.after(0, lambda e=err: self._handle_error(f"Error: {e}"))
        except Exception as exc:
            self.master.after(0, lambda ex=exc: self._handle_error(f"Error during {operation_name}: {ex}"))
        finally:
            self.master.after(0, self.hide_loader)

    def run_operation(self, operation_func, operation_name, **kwargs):
        if self.original_img is None:
            print("Please load an image first.")
            return
        self._start_background_task(
            lambda: self._run_operation_worker(operation_func, operation_name, kwargs),
            f"{operation_name} in progress..."
        )
        
    def run_resize(self):
        if self.original_img is None:
            print("Please load an image first.")
            return

        def resize_worker():
            try:
                w = self.resize_width.get()
                h = self.resize_height.get()
                base_img = self.processed_img if self.processed_img is not None else self.original_img
                resized = apply_resizing(base_img, w, h)
                self.master.after(0, lambda: self._complete_operation(resized, "Resize"))
            except Exception as e:
                self.master.after(0, lambda ex=e: self._handle_error(f"Error during resizing: {ex}"))
            finally:
                self.master.after(0, self.hide_loader)

        self._start_background_task(resize_worker, "Resizing image...")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    
    # Ensure display updates after the window is drawn to get correct panel sizes
    root.update_idletasks() 
    root.mainloop()