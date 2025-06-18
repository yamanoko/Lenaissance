import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageTk
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OCR 前処理アプリ")
        self.geometry("1000x800")

        # 画像状態管理
        self.orig_np = None
        self.processed_np = None
        self.zoom_level = 1.0
        # 領域選択開始座標および矩形ID
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.recognition_predictor = RecognitionPredictor()
        self.detection_predicor = DetectionPredictor()

        # UI 初期化
        self._init_ui()

    def _init_ui(self):
        # 左パネル: ファイル選択＆前処理設定＆ズーム
        self.frame_left = ctk.CTkFrame(self, width=300)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_open = ctk.CTkButton(
            self.frame_left, text="ファイルを選択", command=self.open_file
        )
        self.btn_open.pack(pady=10)

        self.bin_checkbox = ctk.CTkCheckBox(
            self.frame_left, text="2値化を有効", command=self.update_image
        )
        self.bin_checkbox.pack(pady=10)

        self.bin_slider = ctk.CTkSlider(
            self.frame_left,
            from_=0,
            to=255,
            number_of_steps=255,
            command=lambda v: self.update_image(),
        )
        self.bin_slider.set(127)
        self.bin_slider.pack(pady=10)

        # ズーム設定
        ctk.CTkLabel(self.frame_left, text="ズーム:").pack(pady=(20, 5))
        self.zoom_slider = ctk.CTkSlider(
            self.frame_left, from_=0.5, to=3.0, number_of_steps=25, command=self.on_zoom
        )
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(pady=5)

        # 右パネル: スクロール可能な Canvas コンテナ
        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Canvas とスクロールバー
        self.canvas = tk.Canvas(self.frame_right, bg="gray")
        self.h_scroll = tk.Scrollbar(
            self.frame_right, orient="horizontal", command=self.canvas.xview
        )
        self.v_scroll = tk.Scrollbar(
            self.frame_right, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set
        )
        # 配置
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.frame_right.grid_rowconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(0, weight=1)

        # Canvas マウスイベント
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_file(self):
        file_types = [("Image Files", "*.png *.jpg *.jpeg"), ("PDF Files", "*.pdf")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if not file_path:
            return
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            pages = convert_from_path(file_path, dpi=200)
            pil = pages[0]
        else:
            pil = Image.open(file_path)
        # numpy 形式 (BGR)
        np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        self.orig_np = np_img.copy()
        self.processed_np = np_img.copy()
        self.update_image()

    def update_image(self):
        if self.orig_np is None:
            return
        img = self.orig_np.copy()
        if self.bin_checkbox.get():
            thresh = int(self.bin_slider.get())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        self.processed_np = img
        self._display_on_canvas(img)

    def on_zoom(self, value):
        self.zoom_level = float(value)
        if self.processed_np is not None:
            self._display_on_canvas(self.processed_np)

    def _display_on_canvas(self, img: np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 表示用にリサイズ
        w, h = pil.size
        disp_w = int(w * self.zoom_level)
        disp_h = int(h * self.zoom_level)
        # Image.LANCZOS を使用
        pil_disp = pil.resize((disp_w, disp_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_disp)
        self.canvas.delete("all")
        # 画像を Canvas に配置
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        # スクロール領域を更新
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

    def on_mouse_down(self, event):
        self.start_x, self.start_y = (
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_mouse_move(self, event):
        if self.start_x is None or self.start_y is None:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, x, y, outline="blue", width=2
        )

    def on_mouse_up(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        if self.start_x is None or self.start_y is None:
            return
        self._crop_and_recognize(self.start_x, self.start_y, end_x, end_y)
        self.start_x = None
        self.start_y = None

    def _crop_and_recognize(self, x0, y0, x1, y1):
        img = self.processed_np
        # ズームを補正
        ix0 = int(x0 / self.zoom_level)
        iy0 = int(y0 / self.zoom_level)
        ix1 = int(x1 / self.zoom_level)
        iy1 = int(y1 / self.zoom_level)
        crop = img[min(iy0, iy1) : max(iy0, iy1), min(ix0, ix1) : max(ix0, ix1)]
        if crop.size == 0:
            messagebox.showwarning("Warning", "選択領域が小さすぎます")
            return
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # pil_crop.save("crop.png")
        predictions = self.recognition_predictor(
            [pil_crop], det_predictor=self.detection_predicor
        )[0]
        output_text_file = "crop.txt"
        with open(output_text_file, "a", encoding="utf-8") as f:
            for pred in predictions.text_lines:
                f.write(f"{pred.text}\n")
        messagebox.showinfo(
            "Info", "領域を切り出してOCRしたテキストを crop.txt に保存しました"
        )


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = OCRApp()
    app.mainloop()
