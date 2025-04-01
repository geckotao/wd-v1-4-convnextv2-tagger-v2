import argparse
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import csv
import time

# 模型配置
models_dir = "models"

def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models

# 推理类
class TaggerInference:
    def __init__(self, model_name="wd-v1-4-convnextv2-tagger-v2.onnx", csv_path=None):
        if model_name.endswith(".onnx"):
            model_name = model_name[0:-5]
        installed = list(get_installed_models())
        if not any(model_name + ".onnx" in s for s in installed):
            print(f"模型 {model_name} 未安装，需要下载。")
            
        name = os.path.join(models_dir, model_name + ".onnx")
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = os.cpu_count()  # 使用全部CPU核心
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            options.enable_profiling = False  # 关闭性能分析日志
            self.session = ort.InferenceSession(name, providers=["CPUExecutionProvider"], sess_options=options)
            self.tags, self.general_index, self.character_index = self._read_tags(model_name)
        except Exception as e:
            print(f"初始化推理器时出错: {e}")
            raise

    def _read_tags(self, model_name):
        tags = []
        general_index = None
        character_index = None
        with open(os.path.join(models_dir, model_name + ".csv")) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1].replace("_", " "))
        return tags, general_index, character_index

    def predict(self, image_path: str, base_threshold=0.30, character_threshold=0.85) -> (pd.DataFrame, float):
        try:
            start_time = time.time()  # 记录开始时间
            image = Image.open(image_path)
            input = self.session.get_inputs()[0]
            height = input.shape[1]

            # 等比缩放至最大尺寸并填充白色
            ratio = float(height) / max(image.size)
            new_size = tuple([int(x * ratio) for x in image.size])
            image = image.resize(new_size, Image.LANCZOS)
            square = Image.new("RGB", (height, height), (255, 255, 255))
            square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

            image = np.array(square).astype(np.float32)
            image = image[:, :, ::-1]  # RGB -> BGR
            image = np.expand_dims(image, 0)

            label_name = self.session.get_outputs()[0].name
            probs = self.session.run([label_name], {input.name: image})[0]

            result = list(zip(self.tags, probs[0]))

            general = [item for item in result[self.general_index:self.character_index] if item[1] > base_threshold]
            character = [item for item in result[self.character_index:] if item[1] > character_threshold]

            all_tags = character + general

            result_df = pd.DataFrame(all_tags, columns=['tag', 'confidence'])
            end_time = time.time()  # 记录结束时间
            inference_time = end_time - start_time  # 计算推理时间
            return result_df.sort_values('confidence', ascending=False), inference_time
        except Exception as e:
            print(f"推理过程中发生错误: {e}")
            return pd.DataFrame(), 0

# GUI应用
class TaggerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("wd-v1-4-convnextv2-tagger-v2图像反推 by geckotao")
        self.geometry("600x500")

        # 初始化模型
        self.tagger = TaggerInference()

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 控制面板
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_open = ttk.Button(control_frame, text="打开图片", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT)

        self.threshold_var = tk.DoubleVar(value=35)  # 初始值设为35%
        ttk.Scale(control_frame, from_=10, to=90, variable=self.threshold_var,
                  command=lambda v: self.update_threshold_label(v)).pack(side=tk.RIGHT, padx=5)
        self.threshold_label = ttk.Label(control_frame, text="置信度阈值: 35%")
        self.threshold_label.pack(side=tk.RIGHT)

        # 推理时间显示
        self.inference_time_label = ttk.Label(control_frame, text="推理时间: N/A")
        self.inference_time_label.pack(side=tk.LEFT, padx=5)

        # 图像显示
        self.image_label = ttk.Label(self)
        self.image_label.pack(pady=10)

        # 结果文本框
        self.result_text = tk.Text(self, height=5)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.current_image = file_path
            self.update_display()

    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"置信度阈值: {int(float(value))}%")
        self.update_display()

    def update_display(self):
        try:
            # 显示图片
            img = Image.open(self.current_image).resize((300, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)

            # 获取阈值并转换为小数
            threshold = self.threshold_var.get() / 100

            # 执行预测
            results, inference_time = self.tagger.predict(self.current_image, threshold)

            # 更新推理时间显示
            self.inference_time_label.config(text=f"推理时间: {inference_time:.4f} 秒")

            # 清空旧结果
            self.result_text.delete(1.0, tk.END)

            # 获取所有标签名并用逗号连接
            tag_names = ', '.join(results['tag'])

            # 添加新结果到文本框
            self.result_text.insert(tk.END, tag_names)
        except Exception as e:
            print(f"更新显示时出错: {e}")

if __name__ == "__main__":
    app = TaggerApp()
    app.mainloop()