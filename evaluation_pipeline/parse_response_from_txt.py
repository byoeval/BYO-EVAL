import json
import re
from collections import defaultdict
from pathlib import Path


def parse_chess_file(filepath):
    with open(filepath) as file:
        lines = file.readlines()

    result = {
        "board_info": {},
        "cell_positions": {},
        "pieces": defaultdict(list)
    }

    current_piece_type = None
    current_piece = {}

    for line in lines:
        line = line.strip()

        if line.startswith("Board Position:"):
            result["board_info"]["board_position"] = tuple(
                map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
        elif line.startswith("Board Size:"):
            result["board_info"]["board_size"] = tuple(
                map(int, re.findall(r"\d+", line)))
        elif line.startswith("Physical Size:"):
            result["board_info"]["physical_size"] = tuple(
                map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
        elif line.startswith("Pattern:"):
            result["board_info"]["pattern"] = line.split(": ")[1]
        elif line.startswith("White Cell Color:"):
            result["board_info"]["white_color"] = tuple(
                map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
        elif line.startswith("Black Cell Color:"):
            result["board_info"]["black_color"] = tuple(
                map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))

        elif line.startswith("cell_row"):
            match = re.match(
                r"cell_row_(\d+)_col_(\d+): \(([^,]+), ([^)]+)\)", line)
            if match:
                row, col, x, y = int(match[1]), int(
                    match[2]), float(match[3]), float(match[4])
                result["cell_positions"][(row, col)] = (x, y)

        elif line.endswith("PIECES (1)") or re.match(r"[A-Z]+ PIECES \(\d+\)", line):
            if current_piece:
                result["pieces"][current_piece_type].append(current_piece)
                current_piece = {}
            current_piece_type = line.split()[0].lower()

        elif line.startswith("·"):
            if current_piece:
                result["pieces"][current_piece_type].append(current_piece)
            current_piece = {"name": line[2:-1]}

        elif line.startswith("-"):
            key_val = line.split(": ", 1)
            if len(key_val) == 2:
                key = key_val[0].replace(
                    "-", "").strip().lower().replace(" ", "_")
                val = key_val[1]
                if val.startswith("("):
                    current_piece[key] = tuple(
                        map(float, re.findall(r"[-+]?\d*\.\d+|\d+", val)))
                elif val.lower() in ["true", "false"]:
                    current_piece[key] = val.lower() == "true"
                elif re.match(r"^-?\d+\.?\d*$", val):
                    current_piece[key] = float(val) if '.' in val else int(val)
                else:
                    current_piece[key] = val

    if current_piece:
        result["pieces"][current_piece_type].append(current_piece)

    return result


def parse_all_txt_files(folder_path):
    folder = Path(folder_path)
    output = {}

    for txt_file in folder.glob("*.txt"):
        output[txt_file.name] = parse_chess_file(txt_file)

    return output


# Example usage ---> Test sur quelques fichiers
if __name__ == "__main__":
    import os
    import sys
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "./legends_txt_test"
    all_data = parse_all_txt_files(folder_path)
    output_folder = folder_path + "_parsed"
    os.makedirs(output_folder, exist_ok=True)
    # Optional: Save to JSON
    with open(f"{output_folder}/chess_dataset.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Parsed {len(all_data)} files and saved to chess_dataset.json")
