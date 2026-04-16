from flask import Flask, request, render_template, jsonify, send_file # type: ignore
import os
import werkzeug # type: ignore
import uuid
import subprocess
import threading
import time
import re
import warnings
import base64
import numpy as np # type: ignore
import cv2 # type: ignore
from processor import process_image, export_to_excel, export_session_to_excel # type: ignore

# Suppress PyTorch pin_memory warnings globally
warnings.filterwarnings("ignore", message=".*pin_memory.*")

app = Flask(__name__)
mobile_uploads = {}
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/mobile')
def mobile_interface():
    """Mobile capture interface"""
    return render_template('mobile.html')

@app.route('/api/mobile_upload', methods=['POST'])
def mobile_upload():
    """Handle mobile image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
        
        file = request.files['file']
        roll_number = request.form.get('roll_number', 'Unknown')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Use existing upload logic
        session_id = 'default_session'
        
        if file and werkzeug.utils.secure_filename(file.filename):
            filename = werkzeug.utils.secure_filename(file.filename)
            session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(session_folder, exist_ok=True)
            
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            
            # Process image using existing logic
            try:
                results, total, debug_img = process_image(file_path, session_folder)
                
                excel_filename = 'marks_database.xlsx'
                excel_path = os.path.join(session_folder, excel_filename)
                export_to_excel(roll_number, results, total, excel_path)
                
                # Store in memory for laptop polling
                mobile_uploads[roll_number] = {
                    'success': True,
                    'results': results,
                    'total': total,
                    'roll_number': roll_number,
                    'excel_file': excel_filename,
                    'debug_img': debug_img
                }
                
                return jsonify({
                    'success': True,
                    'message': 'Image uploaded and processed successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'})
        
        return jsonify({'success': False, 'error': 'Invalid file'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    roll_number = request.form.get('roll_number', 'Unknown')
    session_id = request.form.get('session_id', 'default_session')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        ext = os.path.splitext(filename)[1]
        unique_id = str(uuid.uuid4())
        unique_filename = unique_id + ext
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        filepath = os.path.join(session_folder, unique_filename)
        file.save(filepath)
        
        try:
            results, total, debug_filename = process_image(filepath)
            
            excel_filename = 'marks_database.xlsx'
            excel_path = os.path.join(session_folder, excel_filename)
            export_to_excel(roll_number, results, total, excel_path)
            
            if request.headers.get('Accept') == 'application/json' or request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'results': results,
                    'total': total,
                    'roll_number': roll_number,
                    'excel_file': excel_filename,
                    'debug_img': debug_filename
                })
            
            return render_template('result.html', results=results, total=total, roll_number=roll_number, excel_file=excel_filename, debug_img=debug_filename)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if request.headers.get('Accept') == 'application/json' or request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': str(e)})
            return jsonify({'error': str(e)})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    session_id = request.form.get('session_id', 'default_session')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            import fitz # type: ignore
        except ImportError:
            return jsonify({'error': 'PyMuPDF (fitz) is not installed'})
            
        filename = werkzeug.utils.secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        pdf_path = os.path.join(session_folder, f"{unique_id}.pdf")
        file.save(pdf_path)
        
        all_results = []
        try:
            doc = fitz.open(pdf_path)
            # HIGH-RES rendering for accurate digit recognition
            # 4.0x = ~288 DPI (much better than the previous 2.0x = 144 DPI)
            mat = fitz.Matrix(4.0, 4.0)
            
            for page_num in range(len(doc)):
                # Page 1 -> Roll 1, etc.
                roll_number = str(page_num + 1).zfill(2) # e.g. 01, 02...
                
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat)
                
                img_filename = f"{unique_id}_page{page_num+1}.jpg"
                img_path = os.path.join(session_folder, img_filename)
                
                # Save as high-quality JPEG
                pix.save(img_path)
                
                # Process image through improved ML pipeline
                results, total, debug_filename = process_image(img_path)
                
                # Update rolling excel sheet
                excel_filename = 'marks_database.xlsx'
                excel_path = os.path.join(session_folder, excel_filename)
                export_to_excel(roll_number, results, total, excel_path)
                
                all_results.append({
                    'roll_number': roll_number,
                    'results': results,
                    'total': total,
                    'debug_img': debug_filename,
                    'excel_file': excel_filename,
                    'success': True
                })
                
            doc.close()
            
            return jsonify({
                'success': True,
                'batch_results': all_results,
                'total_processed': len(all_results)
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file format. Must be PDF.'})

@app.route('/api/webcam_capture', methods=['POST'])
def webcam_capture():
    """Handle webcam frame capture — receives base64 image data."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        roll_number = data.get('roll_number', 'Unknown')
        session_id = data.get('session_id', 'default_session')
        image_data = data['image']
        
        # Decode base64 image
        # Strip header if present (e.g., "data:image/jpeg;base64,...")
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'})
        
        # Save the captured frame
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        unique_id = str(uuid.uuid4())
        img_filename = f"webcam_{unique_id}.jpg"
        img_path = os.path.join(session_folder, img_filename)
        cv2.imwrite(img_path, img)
        
        # Process through the ML pipeline using specific webcam glare-handling parameters
        results, total, debug_filename = process_image(img_path, is_webcam=True)
        
        # Save to Excel
        excel_filename = 'marks_database.xlsx'
        excel_path = os.path.join(session_folder, excel_filename)
        export_to_excel(roll_number, results, total, excel_path)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': total,
            'roll_number': roll_number,
            'excel_file': excel_filename,
            'debug_img': debug_filename
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_marks', methods=['POST'])
def update_marks():
    try:
        data = request.get_json()
        roll_number = data.get('roll_number', 'Unknown')
        results = data.get('results', [])
        excel_file = data.get('excel_file', 'marks_database.xlsx')
        session_id = data.get('session_id', 'default_session')
        
        # Update the Excel file with edited marks
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, excel_file)
        export_to_excel(roll_number, results, sum([int(m['mark']) for m in results if str(m['mark']).isdigit()]), excel_path)
        
        return jsonify({'success': True, 'message': 'Marks updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_edited_marks', methods=['POST'])
def export_edited_marks():
    try:
        data = request.get_json()
        roll_number = data.get('roll_number', 'Unknown')
        results = data.get('results', [])
        session_id = data.get('session_id', 'default_session')
        
        # Create user-specific Excel file
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        export_filename = f'marks_{roll_number}_edited.xlsx'
        export_path = os.path.join(session_folder, export_filename)
        
        export_to_excel(roll_number, results, sum([int(m['mark']) for m in results if str(m['mark']).isdigit()]), export_path)
        
        # Return the strictly formatted file for download directly from the session drive
        return send_file(export_path, as_attachment=True, download_name=export_filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_all_session', methods=['POST'])
def export_all_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default_session')
        
        # Get all history from session data
        session_history = data.get('session_history', [])
        
        if not session_history:
            return jsonify({'success': False, 'error': 'No session data found'})
        
        # Create comprehensive Excel file with all session data
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Generate unique filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f'marks_session_{timestamp}.xlsx'
        export_path = os.path.join(session_folder, export_filename)
        
        # Export all session data to Excel
        export_session_to_excel(session_history, export_path)
        
        # Return the file for download
        return send_file(export_path, as_attachment=True, download_name=export_filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/check_mobile_upload/<roll_number>')
def check_mobile_upload(roll_number):
    """Endpoint for laptop GUI to poll for mobile upload completion"""
    if roll_number in mobile_uploads:
        data = mobile_uploads.pop(roll_number) # Pop to clear it
        return jsonify(data)
    return jsonify({'success': False, 'status': 'waiting'})

@app.route('/api/get_ngrok_url')
def get_ngrok_url():
    """Return the generated ngrok URL for QR codes"""
    return jsonify({'url': app.config.get("BASE_URL", f"http://localhost:5000")})

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], session_id, filename)
    return send_file(filepath, as_attachment=True)

def start_ngrok_tunnel(port):
    """Starts a public tunnel using ngrok"""
    try:
        from pyngrok import ngrok
        print(" * Starting ngrok tunnel...")
        
        # Open a ngrok tunnel to the dev server
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        print(f" * ngrok tunnel active: {public_url}")
        return public_url
    except Exception as e:
        print(" * ngrok failed:", e)
        print("   Note: ngrok may require an authtoken.")
        print("   To fix, create a free account at https://dashboard.ngrok.com")
        print("   Then run: pyngrok config add-authtoken <YOUR_TOKEN>")
        return None

if __name__ == '__main__':
    port = 5000
    try:
        if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            public_url = start_ngrok_tunnel(port)
            if public_url:
                print(f" * Free Tunnel Active \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
                os.environ["NGROK_PUBLIC_URL"] = public_url
            else:
                print(" * Failed to start free tunnels. Falling back to localhost.")
                os.environ["NGROK_PUBLIC_URL"] = f"http://localhost:{port}"
                
        # Set config in both parent and child (reloader) processes
        app.config["BASE_URL"] = os.environ.get("NGROK_PUBLIC_URL", f"http://localhost:{port}")
    except Exception as e:
        print(f"Failed to setup tunnel: {e}")
        app.config["BASE_URL"] = f"http://localhost:{port}"
        
    app.run(host='0.0.0.0', port=port, debug=True)
