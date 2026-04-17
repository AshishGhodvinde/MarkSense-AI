import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

from processor import process_image

GROUND_TRUTH = {
    "images/7.png": {
        "Q1.a": "1",
        "Q1.b": "1",
        "Q1.c": "-",
        "Q1.d": "1",
        "Q1.e": "-",
        "Q1.f": "-",
        "Q2.a": "2",
        "Q2.b": "-",
        "Q3.a": "-",
        "Q3.b": "2",
        "expected_total": 7,
    },
    "images/11.png": {
        "Q1.a": "2",
        "Q1.b": "1",
        "Q1.c": "-",
        "Q1.d": "1",
        "Q1.e": "-",
        "Q1.f": "1",
        "Q2.a": "-",
        "Q2.b": "2",
        "Q3.a": "4",
        "Q3.b": "-",
        "expected_total": 11,
    },
    "images/12.png": {
        "Q1.a": "1",
        "Q1.b": "1",
        "Q1.c": "-",
        "Q1.d": "1",
        "Q1.e": "-",
        "Q1.f": "2",
        "Q2.a": "-",
        "Q2.b": "2",
        "Q3.a": "-",
        "Q3.b": "5",
        "expected_total": 12,
    },
    "images/13.png": {
        "Q1.a": "-",
        "Q1.b": "1",
        "Q1.c": "2",
        "Q1.d": "1",
        "Q1.e": "1",
        "Q1.f": "-",
        "Q2.a": "4",
        "Q2.b": "-",
        "Q3.a": "4",
        "Q3.b": "-",
        "expected_total": 13,
    },
    "images/14.png": {
        "Q1.a": "-",
        "Q1.b": "1",
        "Q1.c": "2",
        "Q1.d": "1",
        "Q1.e": "-",
        "Q1.f": "2",
        "Q2.a": "-",
        "Q2.b": "3",
        "Q3.a": "-",
        "Q3.b": "5",
        "expected_total": 14,
    },
    "images/15.png": {
        "Q1.a": "2",
        "Q1.b": "1",
        "Q1.c": "-",
        "Q1.d": "2",
        "Q1.e": "-",
        "Q1.f": "1",
        "Q2.a": "-",
        "Q2.b": "5",
        "Q3.a": "4",
        "Q3.b": "-",
        "expected_total": 15,
    },
    "images/16.png": {
        "Q1.a": "2",
        "Q1.b": "1",
        "Q1.c": "2",
        "Q1.d": "1",
        "Q1.e": "1",
        "Q1.f": "-",
        "Q2.a": "4",
        "Q2.b": "-",
        "Q3.a": "5",
        "Q3.b": "-",
        "expected_total": 16,
    },
    "images/18.png": {
        "Q1.a": "2",
        "Q1.b": "1",
        "Q1.c": "2",
        "Q1.d": "2",
        "Q1.e": "2",
        "Q1.f": "-",
        "Q2.a": "4",
        "Q2.b": "-",
        "Q3.a": "5",
        "Q3.b": "-",
        "expected_total": 18,
    },
    "images/19.png": {
        "Q1.a": "2",
        "Q1.b": "-",
        "Q1.c": "2",
        "Q1.d": "2",
        "Q1.e": "2",
        "Q1.f": "1",
        "Q2.a": "-",
        "Q2.b": "5",
        "Q3.a": "5",
        "Q3.b": "-",
        "expected_total": 19,
    },
    "images/20.png": {
        "Q1.a": "2",
        "Q1.b": "2",
        "Q1.c": "2",
        "Q1.d": "2",
        "Q1.e": "2",
        "Q1.f": "-",
        "Q2.a": "-",
        "Q2.b": "5",
        "Q3.a": "5",
        "Q3.b": "-",
        "expected_total": 20,
    },
}


def main():
    total_correct = 0
    total_questions = 0
    total_exact_sheets = 0

    for img_path, expected in GROUND_TRUTH.items():
        expected_total = expected["expected_total"]
        question_truth = {key: value for key, value in expected.items() if key != "expected_total"}

        results, total, _ = process_image(img_path)
        predicted = {row["question"]: str(row["mark"]) for row in results}

        correct = 0
        for question, truth in question_truth.items():
            if predicted.get(question) == truth:
                correct += 1

        exact_sheet = correct == len(question_truth) and total == expected_total
        total_correct += correct
        total_questions += len(question_truth)
        total_exact_sheets += int(exact_sheet)

        print(f"\n{img_path}")
        print(f"  Questions: {correct}/{len(question_truth)}")
        print(f"  Total: predicted={total}, expected={expected_total}")
        print(f"  Exact sheet match: {'YES' if exact_sheet else 'NO'}")

    print("\nSummary")
    print(f"  Question accuracy: {total_correct}/{total_questions} ({100 * total_correct / total_questions:.1f}%)")
    print(f"  Exact sheet accuracy: {total_exact_sheets}/{len(GROUND_TRUTH)} ({100 * total_exact_sheets / len(GROUND_TRUTH):.1f}%)")


if __name__ == "__main__":
    main()
