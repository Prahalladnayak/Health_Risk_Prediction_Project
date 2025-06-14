from flask import Flask, render_template, request, Response, session
import numpy as np
import pickle
from datetime import datetime
import io

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Try to import reportlab with fallback
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.colors import HexColor
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Load model and scaler
try:
    model = pickle.load(open('svc_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✅ Model and scaler loaded successfully")
    print(f"Model classes: {model.classes_}")
except Exception as e:
    print(f"❌ Error loading model/scaler: {str(e)}")
    # Create dummy model for fallback
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # Create dummy scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print("⚠️ Using dummy model")

# Health tips database
HEALTH_TIPS = {
    "general": [
        "🥗 Maintain a balanced diet rich in fruits and vegetables",
        "💧 Drink at least 8 glasses of water daily",
        "😴 Get 7-9 hours of quality sleep each night",
        "🧘 Practice stress-reduction techniques like meditation",
        "🩺 Schedule regular health check-ups with your doctor"
    ],
    "risk": [
        "⚠️ Consult a healthcare professional immediately",
        "📈 Monitor your vital signs regularly",
        "💊 Take prescribed medications as directed",
        "🚭 Avoid smoking and limit alcohol consumption",
        "🚨 Create an emergency health action plan"
    ],
    "healthy": [
        "✅ Continue your current healthy habits",
        "🏆 Maintain your exercise routine",
        "📅 Schedule annual health screenings",
        "📚 Stay informed about health best practices",
        "🌟 Celebrate your health achievements!"
    ],
    "bmi": {
        "underweight": "⚖️ Focus on nutrient-dense foods to gain healthy weight",
        "normal": "⚖️ Maintain your healthy weight with balanced nutrition",
        "overweight": "⚖️ Incorporate more physical activity to manage weight",
        "obese": "⚖️ Work with a nutritionist for a personalized weight management plan"
    }
}

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'age': float(request.form['age']),
            'bp': float(request.form['bp']),
            'chol': float(request.form['chol']),
            'hr': float(request.form['hr']),
            'exercise': int(request.form['exercise']),
            'height': float(request.form['height']),
            'weight': float(request.form['weight']),
            'condition': int(request.form['condition']),
            'medication': int(request.form['medication']),
            'gender': int(request.form['gender']),
            'diabetes': int(request.form['diabetes']),
            'smoker': int(request.form['smoker']),
            'name': request.form.get('name', 'Patient')
        }
        
        # Calculate BMI
        height_m = data['height'] / 100
        bmi = data['weight'] / (height_m ** 2)
        
        # DEBUG: Print input data
        print("\n" + "="*50)
        print("RAW INPUT DATA:")
        for key, value in data.items():
            print(f"{key}: {value}")
        print(f"Calculated BMI: {bmi:.2f}")
        
        # Map form values to model features
        features = np.array([
            data['age'],                     # Age
            data['bp'],                       # Blood_Pressure
            data['chol'],                     # Cholesterol
            data['hr'],                       # Heart_Rate
            data['exercise'],                 # Exercise_Level
            bmi,                              # BMI
            data['condition'],                # Previous_Conditions
            data['medication'],               # Medication
            data['gender'],                   # Gender_Male (1=male, 0=female)
            data['diabetes'],                # Diabetes_Yes (1=yes, 0=no)
            data['smoker']                   # Smoker_Yes (1=yes, 0=no)
        ]).reshape(1, -1)
        
        # DEBUG: Print features before scaling
        print("\nFEATURES BEFORE SCALING:")
        print(features)
        
        # Scale features and predict
        scaled_features = scaler.transform(features)
        
        # DEBUG: Print features after scaling
        print("\nFEATURES AFTER SCALING:")
        print(scaled_features)
        
        # Get prediction probability
        prediction_prob = model.predict_proba(scaled_features)[0]
        
        # DEBUG: Print prediction probabilities
        print(f"Prediction probabilities: {prediction_prob}")
        
        # Get index of class 1
        class1_index = list(model.classes_).index(1)
        risk_percentage = int(prediction_prob[class1_index] * 100)
        
        # Determine prediction class
        prediction_class = 1 if risk_percentage >= 50 else 0
        
        # DEBUG: Print final results
        print(f"Risk percentage: {risk_percentage}%")
        print(f"Prediction class: {prediction_class}")
        print("="*50 + "\n")
        
        # Store data in session for report generation
        session['user_data'] = data
        session['bmi'] = round(bmi, 1)
        session['prediction'] = prediction_class
        session['risk_percentage'] = risk_percentage
        
        # Map prediction to text
        result_text = "⚠️ High Risk of Heart Disease" if prediction_class == 1 else "✅ Low Risk of Heart Disease"
        
        return render_template('index.html', 
                              prediction_text=result_text,
                              show_download=True,
                              risk_percentage=risk_percentage)

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def generate_report_content(data, bmi, bmi_category, prediction, risk_percentage):
    """Generate comprehensive health report content"""
    
    # Map numerical values to human-readable text
    exercise_levels = {1: "Low", 2: "Moderate", 3: "High"}
    conditions = {0: "None", 1: "Heart Attack", 2: "Angina", 3: "Stroke"}
    medications = {1: "Low", 2: "Medium", 3: "High"}
    
    # Header section
    report = f"Patient Health Report\n{'='*30}\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"Patient Name: {data.get('name', 'N/A')}\n\n"
    
    # Vital statistics
    report += "Vital Statistics:\n"
    report += f"• Age: {data.get('age', 'N/A')} years\n"
    report += f"• Blood Pressure: {data.get('bp', 'N/A')} mmHg\n"
    report += f"• Cholesterol: {data.get('chol', 'N/A')} mg/dL\n"
    report += f"• Heart Rate: {data.get('hr', 'N/A')} bpm\n"
    report += f"• BMI: {bmi} ({bmi_category.capitalize()})\n"
    report += f"• Exercise Level: {exercise_levels.get(data.get('exercise', 0), 'N/A')}\n\n"
    
    # Medical history
    report += "Medical History:\n"
    report += f"• Previous Condition: {conditions.get(data.get('condition', 0), 'N/A')}\n"
    report += f"• Medication Level: {medications.get(data.get('medication', 0), 'N/A')}\n"
    report += f"• Diabetes: {'Yes' if data.get('diabetes', 0) == 1 else 'No'}\n"
    report += f"• Smoker: {'Yes' if data.get('smoker', 0) == 1 else 'No'}\n\n"
    
    # Prediction result
    report += "Assessment Result:\n"
    report += f"🔍 Risk Level: {risk_percentage}%\n"
    report += "🔍 " + ("⚠️ High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease") + "\n\n"
    
    # Personalized recommendations
    report += "Personalized Recommendations:\n"
    
    # BMI-specific tip
    report += f"• {HEALTH_TIPS['bmi'].get(bmi_category, 'Maintain a healthy lifestyle')}\n"
    
    # Condition-specific tips
    if data.get('diabetes', 0) == 1:
        report += "• 🩺 Monitor blood sugar levels regularly and follow diabetic diet\n"
    if data.get('smoker', 0) == 1:
        report += "• 🚭 Seek support for smoking cessation programs\n"
    if data.get('exercise', 0) == 1:  # Low exercise
        report += "• � Begin a gradual exercise program (start with 15 mins/day)\n"
    
    # Prediction-specific tips
    tip_key = "risk" if prediction == 1 else "healthy"
    for tip in HEALTH_TIPS.get(tip_key, [])[:3]:
        report += f"• {tip}\n"
    
    # General health tips
    report += "\nGeneral Health Tips:\n"
    for tip in HEALTH_TIPS.get('general', []):
        report += f"• {tip}\n"
    
    # Footer
    report += "\n" + "="*30 + "\n"
    report += "Note: This report is generated by an AI system and should not\n"
    report += "replace professional medical advice. Consult your doctor for\n"
    report += "personalized health guidance."
    
    return report

@app.route('/download_report')
def download_report():
    try:
        # Retrieve data from session
        if 'user_data' not in session:
            return "❌ Error: Session expired. Please submit the form again."
            
        data = session['user_data']
        bmi = session.get('bmi', 0)
        prediction = session.get('prediction', 0)
        risk_percentage = session.get('risk_percentage', 0)
        
        # Determine BMI category
        if bmi < 18.5:
            bmi_category = "underweight"
        elif 18.5 <= bmi < 25:
            bmi_category = "normal"
        elif 25 <= bmi < 30:
            bmi_category = "overweight"
        else:
            bmi_category = "obese"
        
        # If PDF is not supported, generate text report
        if not PDF_SUPPORT:
            report_content = generate_report_content(data, bmi, bmi_category, prediction, risk_percentage)
            return Response(
                report_content,
                mimetype="text/plain",
                headers={"Content-Disposition": f"attachment;filename={data.get('name', 'Health')}_Report.txt"}
            )
        
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF object
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Set up styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            textColor=HexColor("#2563eb"),
            spaceAfter=20
        )
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor("#1d4ed8"),
            spaceAfter=10
        )
        normal_style = styles['BodyText']
        
        # Add title
        title = Paragraph("Patient Health Report", title_style)
        elements.append(title)
        
        # Patient info
        elements.append(Paragraph(f"<b>Patient Name:</b> {data.get('name', 'N/A')}", normal_style))
        elements.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        elements.append(Spacer(1, 20))
        
        # Vital statistics section
        elements.append(Paragraph("Vital Statistics", section_style))
        elements.append(Paragraph(f"<b>• Age:</b> {data.get('age', 'N/A')} years", normal_style))
        elements.append(Paragraph(f"<b>• Blood Pressure:</b> {data.get('bp', 'N/A')} mmHg", normal_style))
        elements.append(Paragraph(f"<b>• Cholesterol:</b> {data.get('chol', 'N/A')} mg/dL", normal_style))
        elements.append(Paragraph(f"<b>• Heart Rate:</b> {data.get('hr', 'N/A')} bpm", normal_style))
        elements.append(Paragraph(f"<b>• BMI:</b> {bmi:.1f} ({bmi_category.capitalize()})", normal_style))
        elements.append(Spacer(1, 20))
        
        # Medical history section
        elements.append(Paragraph("Medical History", section_style))
        
        # Map numerical values to human-readable text
        exercise_levels = {1: "Low", 2: "Moderate", 3: "High"}
        conditions = {0: "None", 1: "Heart Attack", 2: "Angina", 3: "Stroke"}
        medications = {1: "Low", 2: "Medium", 3: "High"}
        
        elements.append(Paragraph(f"<b>• Exercise Level:</b> {exercise_levels.get(data.get('exercise', 0), 'N/A')}", normal_style))
        elements.append(Paragraph(f"<b>• Previous Condition:</b> {conditions.get(data.get('condition', 0), 'N/A')}", normal_style))
        elements.append(Paragraph(f"<b>• Medication Level:</b> {medications.get(data.get('medication', 0), 'N/A')}", normal_style))
        elements.append(Paragraph(f"<b>• Diabetes:</b> {'Yes' if data.get('diabetes', 0) == 1 else 'No'}", normal_style))
        elements.append(Paragraph(f"<b>• Smoker:</b> {'Yes' if data.get('smoker', 0) == 1 else 'No'}", normal_style))
        elements.append(Spacer(1, 20))
        
        # Assessment result section
        elements.append(Paragraph("Assessment Result", section_style))
        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        result_color = "#ef4444" if prediction == 1 else "#10b981"
        
        result_style = ParagraphStyle(
            'Result',
            parent=styles['BodyText'],
            fontSize=16,
            textColor=result_color,
            alignment=TA_CENTER,
            spaceAfter=10
        )
        
        elements.append(Paragraph(f"<b>Risk Level:</b> {risk_percentage}%", result_style))
        elements.append(Paragraph(f"<b>{result_text}</b>", result_style))
        elements.append(Spacer(1, 20))
        
        # Personalized recommendations section
        elements.append(Paragraph("Personalized Recommendations", section_style))
        
        # BMI-specific tip
        elements.append(Paragraph(f"• {HEALTH_TIPS['bmi'].get(bmi_category, 'Maintain a healthy lifestyle')}", normal_style))
        
        # Condition-specific tips
        if data.get('diabetes', 0) == 1:
            elements.append(Paragraph("• 🩺 Monitor blood sugar levels regularly and follow diabetic diet", normal_style))
        if data.get('smoker', 0) == 1:
            elements.append(Paragraph("• 🚭 Seek support for smoking cessation programs", normal_style))
        if data.get('exercise', 0) == 1:  # Low exercise
            elements.append(Paragraph("• 🏃 Begin a gradual exercise program (start with 15 mins/day)", normal_style))
        
        # Prediction-specific tips
        tip_key = "risk" if prediction == 1 else "healthy"
        for tip in HEALTH_TIPS.get(tip_key, [])[:3]:
            elements.append(Paragraph(f"• {tip}", normal_style))
        
        elements.append(Spacer(1, 20))
        
        # General health tips section
        elements.append(Paragraph("General Health Tips", section_style))
        for tip in HEALTH_TIPS.get('general', []):
            elements.append(Paragraph(f"• {tip}", normal_style))
        
        elements.append(Spacer(1, 30))
        
        # Footer note
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['BodyText'],
            fontSize=10,
            fontName='Helvetica-Oblique',
            textColor=HexColor("#64748b"),
            alignment=TA_CENTER
        )
        
        footer_text = "Note: This report is generated by an AI system and should not replace professional medical advice. Consult your doctor for personalized health guidance."
        elements.append(Paragraph(footer_text, footer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the value of the BytesIO buffer
        buffer.seek(0)
        return Response(
            buffer,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename={data.get('name', 'Health')}_Report.pdf"}
        )
    
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)