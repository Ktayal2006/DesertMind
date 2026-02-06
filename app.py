import sys
import os
import time  # <--- Added standard time module
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFrame, QGridLayout, QToolBar
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt6.QtCore import Qt, QSize
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision import transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- COLOR PALETTE & STYLE SHEET ---
COLORS = {
    "bg_dark": "#1a1f29",
    "bg_card": "#293241",
    "text_light": "#e0fbfc",
    "accent_blue": "#98c1d9",
    "accent_orange": "#ee6c4d",
    "border": "#3d5a80"
}

# Fixed font family to work on Mac and Windows
QSS = f"""
    QMainWindow {{ background-color: {COLORS['bg_dark']}; }}
    QToolBar {{ background-color: {COLORS['bg_card']}; border-bottom: 1px solid {COLORS['border']}; }}
    QToolButton {{ color: {COLORS['text_light']}; }}
    QFrame#DashboardCard {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 10px;
    }}
    QLabel {{ color: {COLORS['text_light']}; font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif; }}
    QLabel#CardTitle {{
        font-weight: bold; font-size: 14px; margin-bottom: 10px;
        color: {COLORS['accent_blue']};
    }}
    QPushButton {{
        background-color: {COLORS['accent_orange']};
        color: {COLORS['bg_dark']};
        border: none; border-radius: 5px;
        padding: 10px 20px; font-weight: bold; font-size: 14px;
    }}
    QPushButton:hover {{ background-color: {COLORS['accent_blue']}; }}
    QPushButton:disabled {{ background-color: {COLORS['border']}; color: #666; }}
    QLabel#ImageLabel {{
        border: 1px dashed {COLORS['border']};
        border-radius: 5px;
        background-color: #202633;
    }}
"""

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(COLORS['bg_card'])
        for spine in self.axes.spines.values():
            spine.set_color(COLORS['border'])
        self.axes.tick_params(colors=COLORS['text_light'])
        self.axes.xaxis.label.set_color(COLORS['text_light'])
        self.axes.yaxis.label.set_color(COLORS['text_light'])
        super(MplCanvas, self).__init__(self.fig)

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DESERTMIND ANALYTICS | Off-Road Segmentation")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(QSS)
        
        # Smart Device Selection (Supports Mac MPS, CUDA, and CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Mac M1/M2/M3 support
        else:
            self.device = "cpu"
        
        print(f"Running on device: {self.device}")

        self.num_classes = 10
        self.class_names = [f"Class {i}" for i in range(self.num_classes)]
        self.color_palette = np.array([
            [61, 90, 128], [152, 193, 217], [224, 251, 252], [238, 108, 77], [41, 50, 65],
            [100, 140, 180], [200, 80, 60], [120, 200, 120], [180, 120, 180], [250, 250, 250]
        ], dtype=np.uint8)
        
        self.inference_times = []  
        self.init_model()
        self.setup_ui()
        

    def init_model(self):
        self.model = deeplabv3_mobilenet_v3_large(weights="DEFAULT", aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        model_path = "best_deeplab.pth"
        if os.path.exists(model_path):
            try:
                # Map location handles loading CUDA weights on Mac/CPU
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}. Using random weights.")
        else:
            print(f"Model file '{model_path}' not found. Using random weights.")
        self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup_ui(self):
        self.setup_toolbar()
        self.setup_central_widget()

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        title_label = QLabel("  DESERTMIND ANALYTICS  ")
        title_label.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['accent_orange']};")
        toolbar.addWidget(title_label)

    def create_card(self, title):
        card = QFrame()
        card.setObjectName("DashboardCard")
        layout = QVBoxLayout(card)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("CardTitle")
        layout.addWidget(title_lbl)
        return card, layout

    def setup_central_widget(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        main_layout.addLayout(grid_layout)

        # --- 1. Input & Output Comparison Card ---
        io_card, io_layout = self.create_card("INPUT SCENE & SEGMENTATION RESULT")
        
        imgs_layout = QHBoxLayout()
        self.input_lbl = QLabel("No Image Loaded")
        self.input_lbl.setObjectName("ImageLabel")
        self.input_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_lbl.setMinimumSize(400, 300)
        
        self.output_lbl = QLabel("Analysis Not Run")
        self.output_lbl.setObjectName("ImageLabel")
        self.output_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_lbl.setMinimumSize(400, 300)
        
        imgs_layout.addWidget(self.input_lbl)
        imgs_layout.addWidget(self.output_lbl)
        io_layout.addLayout(imgs_layout)

        self.analyze_btn = QPushButton("LOAD & ANALYZE IMAGE")
        self.analyze_btn.clicked.connect(self.analyze_image)
        io_layout.addWidget(self.analyze_btn)
        
        grid_layout.addWidget(io_card, 0, 0, 1, 2)

        # --- 2. Class Distribution Donut Chart ---
        dist_card, dist_layout = self.create_card("TERRAIN COMPOSITION (DONUT)")
        self.dist_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        dist_layout.addWidget(self.dist_canvas)
        grid_layout.addWidget(dist_card, 0, 2)

        # --- 3. Model Confidence Bar Chart ---
        conf_card, conf_layout = self.create_card("AVERAGE CLASS CONFIDENCE")
        self.conf_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        conf_layout.addWidget(self.conf_canvas)
        grid_layout.addWidget(conf_card, 1, 0)

        # --- 4. Inference Time Line Chart ---
        time_card, time_layout = self.create_card("INFERENCE PERFORMANCE HISTORY")
        self.time_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        time_layout.addWidget(self.time_canvas)
        grid_layout.addWidget(time_card, 1, 1, 1, 2)

        self.update_charts(None, None)

    def analyze_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png *.jpeg)")
        if not fname:
            return

        # 1. Load and Preprocess Image
        try:
            orig_img = Image.open(fname).convert("RGB")
            input_tensor = self.preprocess(orig_img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Display Input Image
        pixmap = QPixmap(fname).scaled(self.input_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.input_lbl.setPixmap(pixmap)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("PROCESSING...")
        QApplication.processEvents()

        # 2. Run Inference (Cross-Platform Timing)
        start_time = time.time()  # <--- Works on Mac & Windows
        
        with torch.no_grad():
            output = self.model(input_tensor)["out"]
            
            # Sync if using GPU/MPS
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()
                
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000
        self.inference_times.append(inference_time_ms)
        if len(self.inference_times) > 15:
            self.inference_times.pop(0)

        # 3. Process Output
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        mask_colored = self.color_palette[prediction]
        mask_img = Image.fromarray(mask_colored).resize(orig_img.size, Image.NEAREST)
        blended_img = Image.blend(orig_img, mask_img, alpha=0.6)

        blended_np = np.array(blended_img)
        h, w, c = blended_np.shape
        qimg = QImage(blended_np.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap_out = QPixmap.fromImage(qimg).scaled(self.output_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.output_lbl.setPixmap(pixmap_out)

        # 4. Update Charts
        self.update_charts(prediction, probabilities)

        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("LOAD & ANALYZE IMAGE")

    def update_charts(self, prediction, probabilities):
        # --- Class Distribution Donut Chart ---
        self.dist_canvas.axes.clear()
        if prediction is not None:
            unique, counts = np.unique(prediction, return_counts=True)
            total_pixels = prediction.size
            percentages = counts / total_pixels * 100
            labels = [self.class_names[i] for i in unique]
            colors = [tuple(c/255 for c in self.color_palette[i]) for i in unique]
            
            wedges, texts, autotexts = self.dist_canvas.axes.pie(
                percentages, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=colors, wedgeprops=dict(width=0.4, edgecolor=COLORS['bg_card']),
                textprops={'color': COLORS['text_light']}
            )
            self.dist_canvas.axes.set_title("Class Distribution", color=COLORS['text_light'])
        else:
             self.dist_canvas.axes.text(0.5, 0.5, "No Data", ha='center', va='center', color=COLORS['text_light'])
        self.dist_canvas.draw()

        # --- Model Confidence Bar Chart ---
        self.conf_canvas.axes.clear()
        if probabilities is not None:
            confidences = []
            for i in range(self.num_classes):
                class_prob = probabilities[0, i, :, :].cpu().numpy()
                mask = prediction == i
                if mask.any():
                    avg_conf = np.mean(class_prob[mask])
                else:
                    avg_conf = 0.0
                confidences.append(avg_conf)
            
            x = np.arange(self.num_classes)
            self.conf_canvas.axes.bar(x, confidences, color=COLORS['accent_blue'], alpha=0.7)
            self.conf_canvas.axes.set_xticks(x)
            self.conf_canvas.axes.set_xticklabels(self.class_names, rotation=45, ha='right')
            self.conf_canvas.axes.set_ylim(0, 1.0)
            self.conf_canvas.axes.set_title("Mean Confidence per Class", color=COLORS['text_light'])
        else:
             self.conf_canvas.axes.text(0.5, 0.5, "No Data", ha='center', va='center', color=COLORS['text_light'])
        self.conf_canvas.draw()

        # --- Inference Time Line Chart ---
        self.time_canvas.axes.clear()
        if self.inference_times:
            x = np.arange(1, len(self.inference_times) + 1)
            self.time_canvas.axes.plot(x, self.inference_times, marker='o', color=COLORS['accent_orange'], linewidth=2)
            self.time_canvas.axes.set_xlabel("Inference ID")
            self.time_canvas.axes.set_ylabel("Time (ms)")
            self.time_canvas.axes.set_title("Inference Latency Trend", color=COLORS['text_light'])
            self.time_canvas.axes.grid(True, which='both', color=COLORS['border'], linestyle='--')
        else:
             self.time_canvas.axes.text(0.5, 0.5, "No Data", ha='center', va='center', color=COLORS['text_light'])
        self.time_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())
