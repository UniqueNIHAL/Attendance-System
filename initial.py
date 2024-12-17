import customtkinter as ctk
from tkinter import messagebox
import subprocess
import sys
import main
# Initialize CustomTkinter
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

# Sample data: Teacher IDs, Passwords, and Names
teachers = {
    "T001": {"name": "Ram", "password": "ram123"},
    "T002": {"name": "Bob Smith", "password": "bob456"},
    "T003": {"name": "Carol Williams", "password": "carol789"},
}

# Sample data for subjects and rooms
subjects = ["Mathematics", "Science", "History", "English", "Art"]
rooms = ["Room 101", "Room 102", "Room 103", "Room 104", "Room 105"]

# Function to handle login
def login():
    teacher_id = entry_id.get().strip()
    password = entry_password.get().strip()

    if teacher_id in teachers:
        if password == teachers[teacher_id]["password"]:
            teacher_name = teachers[teacher_id]["name"]
            messagebox.showinfo("Login Successful", f"Welcome, {teacher_name}!")
            # Open the attendance selection window
            open_attendance_window(teacher_name)
            # Optionally, hide or destroy the login window
            app.withdraw()  # Hides the login window
        else:
            messagebox.showerror("Login Failed", "Incorrect password.")
    else:
        messagebox.showerror("Login Failed", "Teacher ID not found.")

# Function to open the attendance selection window
def open_attendance_window(teacher_name):
    attendance_window = ctk.CTkToplevel(app)
    attendance_window.title("Attendance Viewer")
    attendance_window.geometry("500x400")
    attendance_window.resizable(False, False)

    # Center the attendance window on the screen
    window_width = 500
    window_height = 400
    screen_width = attendance_window.winfo_screenwidth()
    screen_height = attendance_window.winfo_screenheight()
    x_coordinate = int((screen_width/2) - (window_width/2))
    y_coordinate = int((screen_height/2) - (window_height/2))
    attendance_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # Function to handle viewing attendance
    def view_attendance():
        som= subject_optionmenu.get()
        rom= room_optionmenu.get()
        if __name__ == "__main__":
            
            main.call(som,rom)
            
    # Frame for attendance selection
    attendance_frame = ctk.CTkFrame(attendance_window, width=450, height=300)
    attendance_frame.pack(pady=30)
    attendance_frame.pack_propagate(False)  # Prevents the frame from resizing to fit its contents

    # Welcome Label
    label_welcome = ctk.CTkLabel(attendance_frame, text=f"Welcome, {teacher_name}", font=ctk.CTkFont(size=20, weight="bold"))
    label_welcome.pack(pady=10)

    # Subject Selection
    label_subject = ctk.CTkLabel(attendance_frame, text="Select Subject:", font=ctk.CTkFont(size=16, weight="bold"))
    label_subject.pack(pady=10)
    subject_optionmenu = ctk.CTkOptionMenu(attendance_frame, values=subjects)
    subject_optionmenu.pack(pady=5, padx=50, fill="x")

    # Room Selection
    label_room = ctk.CTkLabel(attendance_frame, text="Select Room:", font=ctk.CTkFont(size=16, weight="bold"))
    label_room.pack(pady=10)
    room_optionmenu = ctk.CTkOptionMenu(attendance_frame, values=rooms)
    room_optionmenu.pack(pady=5, padx=50, fill="x")

    # View Attendance Button
    button_view = ctk.CTkButton(attendance_frame, text="Mark Attendance", command=view_attendance, width=150, height=40)
    button_view.pack(pady=20)

    # Handle closing the attendance window
    def on_close():
        attendance_window.destroy()
        app.deiconify()  # Shows the login window again

    attendance_window.protocol("WM_DELETE_WINDOW", on_close)

# Create the main application window (Login Window)
app = ctk.CTk()
app.title("Teacher Login")
app.geometry("700x400")  # Updated size for better layout
app.resizable(False, False)

# Center the window on the screen
window_width = 700
window_height = 400
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_coordinate = int((screen_width/2) - (window_width/2))
y_coordinate = int((screen_height/2) - (window_height/2))
app.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Frame for inputs with specific height and width
frame = ctk.CTkFrame(app, width=500, height=300)
frame.pack(pady=50)
frame.pack_propagate(False)  # Prevents the frame from resizing to fit its contents

# Title Label
label_title = ctk.CTkLabel(frame, text="Teacher Login", font=ctk.CTkFont(size=28, weight="bold"))
label_title.pack(pady=12)

# Teacher ID Entry
label_id = ctk.CTkLabel(frame, text="Teacher ID:", font=ctk.CTkFont(size=16, weight="bold"))
label_id.pack(pady=10)
entry_id = ctk.CTkEntry(frame, placeholder_text="Enter your Teacher ID", font=ctk.CTkFont(size=14))
entry_id.pack(pady=5, padx=100, fill="x")

# Password Entry
label_password = ctk.CTkLabel(frame, text="Password:", font=ctk.CTkFont(size=16, weight="bold"))
label_password.pack(pady=10)
entry_password = ctk.CTkEntry(frame, placeholder_text="Enter your Password", show="*", font=ctk.CTkFont(size=14))
entry_password.pack(pady=5, padx=100, fill="x")

# Login Button
button_login = ctk.CTkButton(frame, text="Login", command=login, width=200, height=40)
button_login.pack(pady=20)

# Run the application
app.mainloop()
