import os
import re

def extract_dialogues_from_file(file_path):
    """Extracts dialogue lines from a text file based on specific patterns."""
    dialogues = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Regex pattern for dialogue lines: time + name format
    dialogue_pattern = re.compile(r"^\d{3}:\d{2}:\d{2} \w+:.*")
    
    # Pattern to identify the special three-line segment
    special_lines = [
        "[\n"
        "Download MP3 audio file\n",
    ]
    
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and dialogue_pattern.match(lines[i]):
            cleaned_line = re.sub(r"\[.*?\]", "", lines[i + 1].strip())
            if len(cleaned_line) > 3:
                dialogues.append(cleaned_line)
            i += 2
        elif i + 3 < len(lines) and lines[i] == special_lines[0] and lines[i + 1] == special_lines[1]:
            dialogues.append(lines[i + 3].strip())
            i += 4
        i += 1
    
    return dialogues

def process_apollo_mission_dialogues(mission_id):
    """Processes all text files for a given Apollo mission and extracts dialogues into a single file."""
    mission_dir = f"data/raw/apollo_{mission_id}"
    output_file = f"data/processed/apollo_{mission_id}_dialogues.txt"
    
    if not os.path.exists(mission_dir):
        print(f"Directory not found: {mission_dir}")
        return
    
    all_dialogues = []
    
    for file_name in sorted(os.listdir(mission_dir)):
        file_path = os.path.join(mission_dir, file_name)
        if file_name.endswith(".txt"):
            print(f"Processing: {file_path}")
            dialogues = extract_dialogues_from_file(file_path)
            all_dialogues.extend(dialogues)
    
    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(all_dialogues))
    
    print(f"Saved dialogues to {output_file}")

def process_all_apollo_missions():
    """Processes dialogues for Apollo 7 to Apollo 17."""
    for mission_id in range(7, 18):
        if mission_id < 10:
            mission_id = f"0{mission_id}"
        else:
            mission_id = str(mission_id)
        process_apollo_mission_dialogues(mission_id)

def combine_files(file_dir, output_path):
    """Combines all text files in a directory into a single file."""
    
    with open(output_path, "w", encoding="utf-8") as out_file:
        for file_name in sorted(os.listdir(file_dir)):
            file_path = os.path.join(file_dir, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as in_file:
                    out_file.write(in_file.read())
    
    print(f"Combined dialogues saved to {output_path}")


if __name__ == "__main__":
    process_all_apollo_missions()
