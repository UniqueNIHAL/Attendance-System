import tkinter as tk
from tkinter import ttk
import pandas as pd

# Function to load attendance data
def load_attendance():
    try:
        data = pd.read_csv('attendance.csv')
        return data
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Date", "Time"])

# Function to refresh attendance display
def refresh_attendance():
    attendance_data = load_attendance()
    # Clear the treeview
    for row in tree.get_children():
        tree.delete(row)
    # Populate the treeview with new data
    for index, row in attendance_data.iterrows():
        tree.insert("", "end", values=(row["Name"], row["Date"], row["Time"]))

# Export attendance to a new file
def export_attendance():
    attendance_data = load_attendance()
    export_file = "attendance_export.csv"
    attendance_data.to_csv(export_file, index=False)
    tk.messagebox.showinfo("Export", f"Attendance exported to {export_file}")

# Create main window
root = tk.Tk()
root.title("Attendance Viewer")
root.geometry("600x400")

# Title label
title_label = tk.Label(root, text="Student Attendance", font=("Arial", 18))
title_label.pack(pady=10)

# Treeview to display attendance
columns = ("Name", "Date", "Time")
tree = ttk.Treeview(root, columns=columns, show="headings", height=15)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=150)

tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Scrollbar for the treeview
scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

refresh_button = tk.Button(button_frame, text="Refresh", command=refresh_attendance)
refresh_button.grid(row=0, column=0, padx=10)

export_button = tk.Button(button_frame, text="Export Attendance", command=export_attendance)
export_button.grid(row=0, column=1, padx=10)

# Initial data load
refresh_attendance()

# Run the app
root.mainloop()
