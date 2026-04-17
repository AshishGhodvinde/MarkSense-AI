import os
import shutil

import cv2

from processor import _crop_with_padding, _extract_right_digit_crop, _question_rect, deskew_image, enhance_image, locate_marksheet
from test_sample_marksheets import GROUND_TRUTH


def ensure_clean_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def build_dataset(output_dir: str = "labeled_digits"):
    ensure_clean_dir(output_dir)
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    saved = 0
    for img_path, expected in GROUND_TRUTH.items():
        source = cv2.imread(img_path)
        if source is None:
            continue

        source = deskew_image(source)
        source = locate_marksheet(source)
        source = enhance_image(source)

        for question_name, mark in expected.items():
            if question_name == "expected_total" or mark == "-":
                continue

            left, right, top, bottom = _question_rect(question_name, source.shape[1], source.shape[0])
            crop, _ = _crop_with_padding(source, top, bottom, left, right, 0.03, 0.15)
            digit_crop = _extract_right_digit_crop(crop)
            if digit_crop is None:
                continue

            label = str(int(mark))
            filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{question_name.replace('.', '_')}.png"
            cv2.imwrite(os.path.join(output_dir, label, filename), digit_crop)
            saved += 1

    print(f"Saved {saved} labeled digit crops to '{output_dir}'.")


if __name__ == "__main__":
    build_dataset()
