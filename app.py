import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFrame, QGridLayout, QToolBar, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt6.QtCore import Qt, QSize
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision import transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- TERRAIN CONFIGURATION ---
CLASS_MAP = {
    0: {"name": "Sky", "color": (135, 206, 235)},
    1: {"name": "Sand", "color": (244, 164, 96)},
    2: {"name": "Dry Grass", "color": (189, 183, 107)},
    3: {"name": "Trees", "color": (34, 139, 34)},
    4: {"name": "Bushes", "color": (85, 107, 47)},
    5: {"name": "Rocks", "color": (128, 128, 128)},
    6: {"name": "Logs", "color": (160, 82, 45)},
    7: {"name": "Ground Clutter", "color": (105, 105, 105)},
    8: {"name": "Lush Bushes", "color": (0, 100, 0)},
    9: {"name": "Background", "color": (0, 0, 0)},
}

COLORS = {
    "bg_dark": "#1a1f29",
    "bg_card": "#293241",
    "text_light": "#e0fbfc",
    "accent_blue": "#98c1d9",
    "accent_orange": "#ee6c4d",
    "border": "#3d5a80"
}

QSS = f"""
    QMainWindow {{ background-color: {COLORS['bg_dark']}; }}
    QToolBar {{ background-color: {COLORS['bg_card']}; border-bottom: 1px solid {COLORS['border']}; }}
    QFrame#DashboardCard {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 10px;
    }}
    QLabel {{ color: {COLORS['text_light']}; font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; }}
    QLabel#CardTitle {{ font-weight: bold; font-size: 13px; color: {COLORS['accent_blue']}; text-transform: uppercase; }}
    QPushButton {{
        background-color: {COLORS['accent_orange']};
        color: {COLORS['bg_dark']};
        border-radius: 5px; padding: 12px; font-weight: bold;
    }}
    QPushButton:hover {{ background-color: {COLORS['accent_blue']}; }}
"""

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(COLORS['bg_card'])
        for spine in self.axes.spines.values():
            spine.set_color(COLORS['border'])
        self.axes.tick_params(colors=COLORS['text_light'], labelsize=8)
        super(MplCanvas, self).__init__(self.fig)

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DESERTMIND ANALYTICS | Mission Control")
        self.setMinimumSize(1400, 950)
        self.setStyleSheet(QSS)
        
        # Device Detection
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"

        # Prepare class lists
        self.num_classes = 10
        self.class_names = [CLASS_MAP[i]["name"] for i in range(10)]
        self.color_palette = np.array([CLASS_MAP[i]["color"] for i in range(10)], dtype=np.uint8)
        
        self.inference_times = []  
        self.init_model()
        self.setup_ui()

    def init_model(self):
        # 1. Initialize with aux_loss=True to match your saved file
        self.model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
        
        # 2. Update both classifiers to match your 10 classes
        self.model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(10, self.num_classes, kernel_size=1)
        
        model_path = "best_deeplab.pth"
        if os.path.exists(model_path):
            try:
                # 3. Load the weights
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model loaded successfully with Auxiliary weights.")
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
        # Toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        title = QLabel("  🌵 DESERTMIND ANALYTICS ")
        title.setStyleSheet(f"font-weight: bold; color: {COLORS['accent_orange']}; font-size: 18px;")
        toolbar.addWidget(title)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        grid = QGridLayout()
        grid.setSpacing(15)

        # 1. Main Viewport
        io_card, io_layout = self.create_card("TERRAIN SEGMENTATION VIEW")
        imgs_layout = QHBoxLayout()
        self.input_lbl = self.create_img_placeholder("SOURCE")
        self.output_lbl = self.create_img_placeholder("MASKED")
        imgs_layout.addWidget(self.input_lbl)
        imgs_layout.addWidget(self.output_lbl)
        io_layout.addLayout(imgs_layout)
        
        self.analyze_btn = QPushButton("IMPORT & ANALYZE OFF-ROAD SCENE")
        self.analyze_btn.clicked.connect(self.analyze_image)
        io_layout.addWidget(self.analyze_btn)
        grid.addWidget(io_card, 0, 0, 1, 2)

        # 2. Donut Chart
        dist_card, dist_layout = self.create_card("TERRAIN COMPOSITION")
        self.dist_canvas = MplCanvas(self)
        dist_layout.addWidget(self.dist_canvas)
        grid.addWidget(dist_card, 0, 2)

        # 3. Bar Chart
        conf_card, conf_layout = self.create_card("MODEL CONFIDENCE")
        self.conf_canvas = MplCanvas(self)
        conf_layout.addWidget(self.conf_canvas)
        grid.addWidget(conf_card, 1, 0)

        # 4. Latency Chart
        time_card, time_layout = self.create_card("PERFORMANCE TREND")
        self.time_canvas = MplCanvas(self)
        time_layout.addWidget(self.time_canvas)
        grid.addWidget(time_card, 1, 1)

        # 5. LEGEND CARD (Your requested mapping)
        legend_card, legend_layout = self.create_card("TERRAIN COLOR KEY")
        legend_grid = QGridLayout()
        for i in range(10):
            color = CLASS_MAP[i]["color"]
            label = CLASS_MAP[i]["name"]
            
            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: rgb{color}; border-radius: 3px;")
            
            name = QLabel(label)
            name.setStyleSheet("font-size: 11px; font-weight: bold;")
            
            legend_grid.addWidget(swatch, i % 5, (i // 5) * 2)
            legend_grid.addWidget(name, i % 5, (i // 5) * 2 + 1)
        
        legend_layout.addLayout(legend_grid)
        grid.addWidget(legend_card, 1, 2)

        main_layout.addLayout(grid)
        self.update_charts(None, None)

    def create_card(self, title):
        card = QFrame(); card.setObjectName("DashboardCard")
        layout = QVBoxLayout(card)
        t = QLabel(title); t.setObjectName("CardTitle")
        layout.addWidget(t)
        return card, layout

    def create_img_placeholder(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(400, 300)
        lbl.setStyleSheet(f"border: 1px dashed {COLORS['border']}; background: #1a1f29;")
        return lbl

    def analyze_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image')
        if not fname: return

        orig_img = Image.open(fname).convert("RGB")
        input_tensor = self.preprocess(orig_img).unsqueeze(0).to(self.device)

        # Update Input UI
        self.input_lbl.setPixmap(QPixmap(fname).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio))
        QApplication.processEvents()

        # Inference
        start = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)["out"]
            if self.device == "mps": torch.mps.synchronize()
        end = time.time()

        self.inference_times.append((end-start)*1000)
        
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Color processing
        mask_colored = self.color_palette[pred]
        mask_img = Image.fromarray(mask_colored).resize(orig_img.size, Image.NEAREST)
        blended = Image.blend(orig_img, mask_img, alpha=0.55)

        # Show Result
        res = np.array(blended)
        qimg = QImage(res.data, res.shape[1], res.shape[0], res.shape[1]*3, QImage.Format.Format_RGB888)
        self.output_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio))

        self.update_charts(pred, probs)

    def update_charts(self, pred, probs):
        # Donut Chart
        self.dist_canvas.axes.clear()
        if pred is not None:
            ids, counts = np.unique(pred, return_counts=True)
            labels = [CLASS_MAP[i]["name"] for i in ids]
            clrs = [tuple(c/255 for c in CLASS_MAP[i]["color"]) for i in ids]
            self.dist_canvas.axes.pie(counts, labels=labels, colors=clrs, wedgeprops=dict(width=0.4), textprops={'color': 'white', 'fontsize': 7})
        self.dist_canvas.draw()

        # Bar Chart
        self.conf_canvas.axes.clear()
        if probs is not None:
            confs = [np.mean(probs[0,i].cpu().numpy()[pred==i]) if (pred==i).any() else 0 for i in range(10)]
            self.conf_canvas.axes.bar(self.class_names, confs, color=COLORS['accent_blue'])
            self.conf_canvas.axes.tick_params(axis='x', rotation=45)
        self.conf_canvas.draw()

        # Time Chart
        self.time_canvas.axes.clear()
        if self.inference_times:
            self.time_canvas.axes.plot(self.inference_times, color=COLORS['accent_orange'], marker='o')
        self.time_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())
