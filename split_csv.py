import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
def split_csv(input_file, num_parts):
    # Read all rows from the input CSV (assume only one column)
    file = Path(input_file)
    with file.open(newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    total_rows = len(rows)
    part_size = (total_rows + num_parts - 1) // num_parts  # ceil division

    for i in range(num_parts):
        part_rows = rows[i*part_size : (i+1)*part_size]
        output_file = file.parent / Path(file.stem) / Path(f'{file.stem}_part_{i+1}.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        with output_file.open('w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(part_rows)


class SplitCSVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Splitter")
        self.geometry("500x160")  # Increased width and height
        self.resizable(False, False)

        # File path
        tk.Label(self, text="CSV File:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.file_var = tk.StringVar()
        self.file_entry = tk.Entry(self, textvariable=self.file_var, width=40)  # Wider entry
        self.file_entry.grid(row=0, column=1, padx=5, sticky="we")
        tk.Button(self, text="Browse...", command=self.browse_file, width=10).grid(row=0, column=2, padx=5)

        # Number of parts
        tk.Label(self, text="Number of parts:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.parts_var = tk.StringVar()
        self.parts_entry = tk.Entry(self, textvariable=self.parts_var, width=10)
        self.parts_entry.grid(row=1, column=1, sticky="w", padx=5)

        # Split button
        tk.Button(self, text="Split", command=self.run_split).grid(row=2, column=1, pady=20)

        self.grid_columnconfigure(1, weight=1)  # Allow column 1 to expand

    def browse_file(self):
        file = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        if file:
            self.file_var.set(file)

    def run_split(self):
        file = self.file_var.get()
        try:
            num_parts = int(self.parts_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of parts.")
            return

        if not file:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        if num_parts < 1:
            messagebox.showerror("Error", "Number of parts must be at least 1.")
            return

        try:
            split_csv(file, num_parts)
            messagebox.showinfo("Success", "CSV split successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    app = SplitCSVApp()
    app.mainloop()