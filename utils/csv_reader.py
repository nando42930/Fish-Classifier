import csv

input_csv = r"C:\Users\Nandj\Downloads\0039026-251025141854904\multimedia.csv"

with open(input_csv, newline='', encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")

    for i, row in enumerate(reader):
        print(i, row)
        if i == 10:
            break
