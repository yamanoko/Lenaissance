import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageTk
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OCR 前処理アプリ")
        self.geometry("1000x800")

        # ページアイテム: (PIL Image, base_name, page_num)
        self.items = []
        self.page_values = []
        self.current_index = None

        # 画像状態管理
        self.orig_np = None
        self.processed_np = None
        self.zoom_level = 1.0
        # 領域選択開始座標および矩形ID
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

        # UI 初期化
        self._init_ui()

    def _init_ui(self):
        # 左パネル: ファイル選択＆前処理設定＆ズーム＆ページ切替＆マージ
        self.frame_left = ctk.CTkFrame(self, width=300)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_open = ctk.CTkButton(
            self.frame_left, text="ファイルを選択", command=self.open_file
        )
        self.btn_open.pack(pady=10)

        self.page_selector = ctk.CTkComboBox(
            self.frame_left, values=self.page_values, command=self.on_page_selected
        )
        self.page_selector.pack(pady=10)

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

        # マージボタン
        self.btn_merge = ctk.CTkButton(
            self.frame_left, text="テキストを結合", command=self.merge_texts
        )
        self.btn_merge.pack(pady=20)

        # 右パネル: スクロール可能な Canvas
        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

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

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.frame_right.grid_rowconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(0, weight=1)

        # Canvas イベント
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_file(self):
        file_types = [("Image Files", "*.png *.jpg *.jpeg"), ("PDF Files", "*.pdf")]
        paths = filedialog.askopenfilenames(filetypes=file_types)
        if not paths:
            return
        # 既存データをクリア
        self.items.clear()
        self.page_values.clear()

        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                # PyMuPDF で PDF ページを画像に変換
                doc = fitz.open(path)
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    mode = "RGB" if pix.alpha == 0 else "RGBA"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    self.items.append((img, base, i + 1))
            else:
                img = Image.open(path)
                self.items.append((img, base, 1))

        # ページセレクター設定
        for pil, base, pnum in self.items:
            label = f"{base}_page{pnum}"
            self.page_values.append(label)
        self.page_selector.configure(values=self.page_values)
        # 初期表示
        self.page_selector.set(self.page_values[0])
        self.on_page_selected(self.page_values[0])

    def on_page_selected(self, label):
        idx = self.page_values.index(label)
        self.current_index = idx
        pil, base, pnum = self.items[idx]
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
        w, h = pil.size
        disp_w = int(w * self.zoom_level)
        disp_h = int(h * self.zoom_level)
        pil_disp = pil.resize((disp_w, disp_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
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
        self._crop_and_save(self.start_x, self.start_y, end_x, end_y)
        self.start_x = None
        self.start_y = None

    def _crop_and_save(self, x0, y0, x1, y1):
        img = self.processed_np
        ix0 = int(x0 / self.zoom_level)
        iy0 = int(y0 / self.zoom_level)
        ix1 = int(x1 / self.zoom_level)
        iy1 = int(y1 / self.zoom_level)
        crop = img[min(iy0, iy1) : max(iy0, iy1), min(ix0, ix1) : max(ix0, ix1)]
        if crop.size == 0:
            messagebox.showwarning("Warning", "選択領域が小さすぎます")
            return
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # ビジーカーソル表示
        self.config(cursor="watch")
        self.update()
        preds = self.recognition_predictor(
            [pil_crop], det_predictor=self.detection_predictor
        )[0]
        self.config(cursor="")
        self.update()
        _, base, pnum = self.items[self.current_index]
        txt_name = f"{base}_page{pnum}.txt"
        with open(txt_name, "w", encoding="utf-8") as f:
            for line in preds.text_lines:
                f.write(f"{line.text}\n")
        messagebox.showinfo(
            "Info", f"ページ {pnum} を OCR して {txt_name} に保存しました"
        )

    def merge_texts(self):
        if not self.items:
            return
        base = self.items[0][1]
        merged_name = f"{base}_all.txt"
        with open(merged_name, "w", encoding="utf-8") as fw:
            for _, b, pnum in self.items:
                txt_name = f"{b}_page{pnum}.txt"
                if os.path.exists(txt_name):
                    with open(txt_name, "r", encoding="utf-8") as fr:
                        fw.write(fr.read())
                        fw.write("\n")
        messagebox.showinfo("Info", f"すべてのテキストを {merged_name} に結合しました")


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = OCRApp()
    app.mainloop()
