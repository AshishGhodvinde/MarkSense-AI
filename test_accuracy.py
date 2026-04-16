import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from processor import process_image
import os

# Ground truth for test images
ground_truths = {
    'images/1.jpeg': {
        'Q1.a': '2', 'Q1.b': '-', 'Q1.c': '1', 'Q1.d': '1',
        'Q1.e': '2', 'Q1.f': '-', 'Q2.a': '-', 'Q2.b': '5',
        'Q3.a': '2', 'Q3.b': '-',
        'expected_total': 13
    },
    'images/16.jpeg': {
        'Q1.a': '2', 'Q1.b': '-', 'Q1.c': '2', 'Q1.d': '1',
        'Q1.e': '2', 'Q1.f': '-', 'Q2.a': '-', 'Q2.b': '5',
        'Q3.a': '4', 'Q3.b': '-',
        'expected_total': 16
    },
}

print("=" * 60)
print("MARKSENSE AI - ACCURACY TEST SUITE")
print("=" * 60)

total_correct = 0
total_questions = 0

for img_path, gt in ground_truths.items():
    if not os.path.exists(img_path):
        print(f"\nSKIPPING {img_path} (file not found)")
        continue
    
    print(f"\n--- Testing: {img_path} ---")
    expected_total = gt.pop('expected_total')
    
    results, total, debug = process_image(img_path)
    
    print(f"{'Question':<10} {'Expected':<10} {'Got':<10} {'Match'}")
    print('-' * 45)
    
    correct = 0
    for r in results:
        q = r['question']
        expected = gt.get(q, '?')
        got = r['mark']
        match = 'OK' if expected == got else 'MISS'
        if expected == got:
            correct += 1
        print(f"{q:<10} {expected:<10} {got:<10} {match}")
    
    total_correct += correct
    total_questions += len(results)
    
    print(f"\nAccuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Expected Total: {expected_total}, Got Total: {total}")
    gt['expected_total'] = expected_total  # restore

print("\n" + "=" * 60)
print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({100*total_correct/total_questions:.1f}%)")
print("=" * 60)

# Also test the PDF if it exists
pdf_path = 'images/marksheets.pdf'
if os.path.exists(pdf_path):
    print(f"\n\n--- Testing PDF: {pdf_path} ---")
    try:
        import fitz
        doc = fitz.open(pdf_path)
        print(f"PDF has {len(doc)} pages")
        
        # Process first 2 pages as a quick test
        mat = fitz.Matrix(4.0, 4.0)
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            
            img_path_tmp = f'images/test_pdf_page{page_num+1}.jpg'
            pix.save(img_path_tmp)
            
            results, total, debug = process_image(img_path_tmp)
            print(f"\n  Page {page_num+1} (Roll {str(page_num+1).zfill(2)}): Total = {total}")
            for r in results:
                print(f"    {r['question']}: {r['mark']}")
            
            # Clean up
            os.remove(img_path_tmp)
        
        doc.close()
    except Exception as e:
        print(f"PDF test failed: {e}")
        import traceback
        traceback.print_exc()
