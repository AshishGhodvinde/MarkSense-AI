# MarkSense AI

MarkSense AI is a Flask-based marksheet digit extraction system for fixed-format handwritten evaluation sheets.

It supports:

- Single image upload
- PDF batch upload where page number maps to roll number
- Webcam capture
- Manual review and correction before export
- Excel export for teacher workflows

## How It Works

The current pipeline is built for a fixed marksheet structure.

1. Normalize the marksheet image with deskewing and perspective correction.
2. Crop fixed `Obt.` cells for each question.
3. Clean the crop and isolate the handwritten score region.
4. Predict the mark using a constrained multi-stage recognizer:
   - real marksheet digit template matching
   - fine-tuned CNN digit recognition
   - EasyOCR fallback
5. Sum question marks to compute the total.
6. Export to Excel.

## Project Files

- [app.py](/A:/Programming/Projects/MarkSense%20AI/app.py): Flask app and upload/export routes
- [processor.py](/A:/Programming/Projects/MarkSense%20AI/processor.py): OCR/CV pipeline
- [train.py](/A:/Programming/Projects/MarkSense%20AI/train.py): base MNIST CNN
- [train_handwritten_marks.py](/A:/Programming/Projects/MarkSense%20AI/train_handwritten_marks.py): fine-tuning on real marksheet digit crops
- [build_labeled_digits.py](/A:/Programming/Projects/MarkSense%20AI/build_labeled_digits.py): create labeled digit crops from sample sheets
- [test_sample_marksheets.py](/A:/Programming/Projects/MarkSense%20AI/test_sample_marksheets.py): sample-sheet regression suite

## Setup

Use Python 3.11 or 3.12.

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

## Evaluate The Pipeline

Run the sample-sheet regression suite:

```powershell
python test_sample_marksheets.py
```

This checks end-to-end extraction on the sample marksheets under `/images`.

## Fine-Tune On Real Marksheet Digits

Build labeled crops from the included sample sheets:

```powershell
python build_labeled_digits.py
```

This creates:

- `labeled_digits/0`
- `labeled_digits/1`
- ...
- `labeled_digits/9`

Then fine-tune the recognizer:

```powershell
python train_handwritten_marks.py
```

The fine-tuned weights are saved to:

- `handwritten_marks_model.pth`
- `mnist_model.pth`

## Recommended Next Improvements

- Add more labeled crops from real teacher marksheets under `labeled_digits/`
- Add at least one clean reference template marksheet for calibration
- Collect webcam-specific samples and fine-tune on those separately
- Add blank-cell examples for explicit filled-vs-empty classification
- Add support for multiple fixed marksheet formats if needed

## Important Notes

- The system currently assumes one fixed marksheet layout.
- Marks are treated as fixed-format values like `01`, `02`, `03`, `04`, `05`, with output exported as numeric values like `1`, `2`, `3`, `4`, `5`.
- Total is computed from extracted question marks rather than OCR of the total row.
