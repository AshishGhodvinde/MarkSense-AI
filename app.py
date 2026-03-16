from flask import Flask, request, render_template, jsonify, send_file # type: ignore
import os
import werkzeug # type: ignore
import uuid
from processor import process_image, export_to_excel # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

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

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], session_id, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
