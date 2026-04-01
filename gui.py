import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

from perceptron import MLP
from dataset import X as DEFAULT_X, y as DEFAULT_Y, normalize, save_dataset, load_dataset


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Нейроэмулятор многослойного персептрона")
        self.root.geometry("760x620")

        self.model = None
        self.errors = []
        self.X = DEFAULT_X
        self.y = DEFAULT_Y

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        row = 0

        def add_label(text):
            nonlocal row
            tk.Label(frame, text=text, anchor="w").grid(row=row, column=0, sticky="w", pady=3)
            row += 1

        def add_entry(default):
            nonlocal row
            e = tk.Entry(frame, width=35)
            e.insert(0, default)
            e.grid(row=row - 1, column=1, sticky="w", padx=8)
            return e

        add_label("Скорость обучения (lr)")
        self.lr_entry = add_entry("0.1")

        add_label("Количество эпох")
        self.epochs_entry = add_entry("1500")

        add_label("Скрытые слои (например: 12,8)")
        self.layers_entry = add_entry("12")

        add_label("Тип активации (sigmoid / tanh / relu)")
        self.activation_combo = ttk.Combobox(frame, values=["sigmoid", "tanh", "relu"], width=32, state="readonly")
        self.activation_combo.set("sigmoid")
        self.activation_combo.grid(row=row - 1, column=1, sticky="w", padx=8)

        add_label("Чувствительность (beta)")
        self.beta_entry = add_entry("1.0")

        add_label("Порог уверенности")
        self.threshold_entry = add_entry("0.5")

        add_label("Инерционность (momentum)")
        self.inertia_entry = add_entry("0.0")

        add_label("Способ коррекции весов")
        self.correction_combo = ttk.Combobox(frame, values=["gradient", "momentum"], width=32, state="readonly")
        self.correction_combo.set("gradient")
        self.correction_combo.grid(row=row - 1, column=1, sticky="w", padx=8)

        row += 1

        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=10)

        tk.Button(btn_frame, text="Загрузить датасет", command=self.load_dataset, width=18).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(btn_frame, text="Сохранить датасет", command=self.save_dataset_file, width=18).grid(row=0, column=1, padx=4, pady=4)
        tk.Button(btn_frame, text="Обучить", command=self.train, width=18).grid(row=0, column=2, padx=4, pady=4)
        tk.Button(btn_frame, text="Показать граф ошибки", command=self.show_plot, width=18).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(btn_frame, text="Сохранить модель", command=self.save_model, width=18).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(btn_frame, text="Загрузить модель", command=self.load_model, width=18).grid(row=1, column=2, padx=4, pady=4)
        tk.Button(btn_frame, text="Предсказание 1-го примера", command=self.predict_first, width=28).grid(row=2, column=0, columnspan=2, padx=4, pady=4)

        row += 1

        tk.Label(frame, text="Лог / вывод").grid(row=row, column=0, sticky="w")
        row += 1

        self.log = tk.Text(frame, height=18, width=88)
        self.log.grid(row=row, column=0, columnspan=2, sticky="nsew")
        frame.grid_rowconfigure(row, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        self.write_log("Готово. Загрузи датасет или используй dataset.py.")

    def write_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def get_dataset(self):
        if self.X is None or self.y is None:
            raise ValueError("Датасет не загружен")
        return np.array(self.X, dtype=float), np.array(self.y, dtype=float)

    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Выберите файл датасета",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            self.X, self.y = load_dataset(path)
            self.write_log(f"Датасет загружен: {path}")
            self.write_log(f"X: {self.X.shape}, y: {self.y.shape}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_dataset_file(self):
        try:
            X_data, y_data = self.get_dataset()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить датасет",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return

        try:
            save_dataset(path, X_data, y_data)
            self.write_log(f"Датасет сохранён: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def build_model(self):
        X_data, y_data = self.get_dataset()
        hidden_layers = [int(x.strip()) for x in self.layers_entry.get().split(",") if x.strip()]

        self.model = MLP(
            input_size=X_data.shape[1],
            hidden_layers=hidden_layers,
            output_size=y_data.shape[1],
            lr=float(self.lr_entry.get()),
            activation=self.activation_combo.get(),
            beta=float(self.beta_entry.get()),
            inertia=float(self.inertia_entry.get()),
            correction=self.correction_combo.get(),
            threshold=float(self.threshold_entry.get())
        )

    def train(self):
        try:
            X_data, y_data = self.get_dataset()
            X_norm = normalize(X_data)

            self.build_model()
            epochs = int(self.epochs_entry.get())

            self.write_log("Начало обучения...")
            self.errors = self.model.train(X_norm, y_data, epochs=epochs, verbose=True)
            self.write_log("Обучение завершено.")

            # Визуализация процесса функционирования: покажем пример предсказания
            pred = self.model.predict(X_norm[:1])[0]
            cls, conf, _ = self.model.predict_class(X_norm[:1])
            self.write_log(f"Выход сети для 1-го примера: {np.round(pred, 4)}")
            self.write_log(f"Класс 1-го примера: {int(cls[0])}, уверенность: {float(conf[0]):.4f}")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_plot(self):
        if not self.errors:
            messagebox.showinfo("Информация", "Сначала обучите сеть.")
            return
        plt.figure()
        plt.plot(self.errors)
        plt.title("График ошибки обучения")
        plt.xlabel("Эпоха")
        plt.ylabel("Средняя абсолютная ошибка")
        plt.grid(True)
        plt.show()

    def save_model(self):
        if self.model is None:
            messagebox.showinfo("Информация", "Сначала обучите или загрузите модель.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить модель",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return

        try:
            self.model.save(path)
            self.write_log(f"Модель сохранена: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Загрузить модель",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return

        try:
            self.model = MLP.load(path)
            self.write_log(f"Модель загружена: {path}")
            self.write_log(f"Архитектура: {self.model.layers}")
            self.write_log(f"Активция: {self.model.activation_name}, lr={self.model.lr}, beta={self.model.beta}, inertia={self.model.inertia}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def predict_first(self):
        try:
            X_data, _ = self.get_dataset()
            if self.model is None:
                raise ValueError("Сначала обучи или загрузи модель.")
            X_norm = normalize(X_data)
            pred = self.model.predict(X_norm[:1])[0]
            cls, conf, _ = self.model.predict_class(X_norm[:1])
            self.write_log("Предсказание для 1-го примера")
            self.write_log(f"Вектор выхода: {np.round(pred, 4)}")
            self.write_log(f"Класс: {int(cls[0])}, уверенность: {float(conf[0]):.4f}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()