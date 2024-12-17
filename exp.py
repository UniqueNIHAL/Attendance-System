import csv

# File name for the CSV
filename = "names.csv"

# Open the file in append mode to add data without overwriting
with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write the header if the file is empty (only if running for the first time)
    file.seek(0, 2)  # Move the cursor to the end of the file
    if file.tell() == 0:
        writer.writerow(["Name"])

    # Input name and write to the file
    while True:
        name = input("Enter a name (or type 'exit' to stop): ").strip()
        if name.lower() == 'exit':
            break
        writer.writerow([name])
        print(f"Name '{name}' has been added to the CSV file.")

print(f"All names saved to {filename}.")
