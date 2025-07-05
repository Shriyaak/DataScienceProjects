import os
import shutil
import zipfile
import re

DATA_DIR = os.path.join(os.getcwd(), "data")

def get_gestures():
      return ['circle', 'come', 'go', 'wave']

def get_columns():
    return ['time', 'accel_x', 'accel_y', 'accel_z', 'accel_abs']

def get_gesture_csvs(gesture):
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    gesture_dir = os.path.join(DATA_DIR, gesture)
    return [file for file in os.listdir(gesture_dir) if file.endswith('.csv')]

def find_highest_numbered_file(directory, full_path=False):
    highest_number = -1
    file_with_highest_number = None

    for entry in os.scandir(directory):
        if entry.is_file():
            # Find all numbers in the filename
            numbers = re.compile(r'\d+').findall(entry.name)
            if numbers:
                # Convert found strings to integers and get the max
                max_number_in_file = max(map(int, numbers))
                if max_number_in_file > highest_number:
                    highest_number = max_number_in_file
                    file_with_highest_number = entry.path

    return file_with_highest_number if full_path else 1 if highest_number == -1 else highest_number

def process_zip(gesture):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    unprocessed_dir = os.path.join(gesture_dir, "unprocessed")
    processed_dir = os.path.join(gesture_dir, "processed")

    file_index = find_highest_numbered_file(gesture_dir)

    for filename in os.listdir(unprocessed_dir): 
        if filename.endswith('.zip'):

            zip_file_path = os.path.join(unprocessed_dir, filename)
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if 'Raw Data.csv' in zip_ref.namelist():
                    with zip_ref.open('Raw Data.csv') as raw_data:
                        content = raw_data.read()
                        new_file_path = os.path.join(DATA_DIR, gesture, f"{file_index}.csv")
                        
                        with open(new_file_path, 'wb') as new_file:
                            new_file.write(content)
                file_index += 1
            
            
            shutil.move(zip_file_path, processed_dir)

def process_all_zips():
    for gesture in get_gestures():
        if os.path.isdir(os.path.join(DATA_DIR, gesture)):

            if len(os.listdir(os.path.join(DATA_DIR, gesture, "unprocessed"))) > 0:
                process_zip(gesture)
                print(f"Extracted all {gesture} gestures to {os.path.join(DATA_DIR, gesture)}")    
            else:
                print(f"No unprocessed files found for {gesture}")

process_all_zips()