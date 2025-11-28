import csv
import os
import requests

input_csv = r"C:\Users\Nandj\Downloads\0039026-251025141854904\multimedia.csv"
output_folder = "boops boops"
os.makedirs(output_folder, exist_ok=True)

image_exts = (".jpg", ".jpeg", ".png")

with open(input_csv, newline='', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")  # IMPORTANT: GBIF uses TAB delimiter

    next(reader)  # skip header

    for row_number, row in enumerate(reader):
        if len(row) <= 3:
            continue

        url = row[3].strip()  # column 4 = identifier

        if not url.lower().endswith(image_exts):
            continue

        # extract photo ID and extension
        photo_id = url.split("/")[-2]  # folder name
        ext = os.path.splitext(url)[1]  # .jpg / .jpeg / .png

        filename = os.path.join(output_folder, photo_id + ext)
        print(f"Row {row_number}: Downloading {url}")

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(filename, "wb") as img:
                img.write(r.content)
        except Exception as e:
            print("Failed:", url, e)

print("Finished!")
