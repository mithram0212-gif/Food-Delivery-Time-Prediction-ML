import tkinter as tk
from tkinter import messagebox

def predict():
    distance = distance_entry.get()

    if distance == "":
        messagebox.showerror("Error", "Enter Distance")
    else:
        delivery_time = float(distance) * 5
        result_label.config(
            text=f"Estimated Delivery Time: {delivery_time:.1f} minutes"
        )

root = tk.Tk()
root.title("Food Delivery Time Prediction")
root.geometry("400x250")

tk.Label(root, text="Distance (km)").pack(pady=10)

distance_entry = tk.Entry(root)
distance_entry.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=15)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()