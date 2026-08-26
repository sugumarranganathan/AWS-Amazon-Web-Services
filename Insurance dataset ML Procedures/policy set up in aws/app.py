from flask import Flask, render_template, request, jsonify
import boto3
import json
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

app = Flask(__name__, template_folder='templates')


# 🔐 Securely fetch AWS credentials from Secrets Manager
def get_aws_secrets():
    secret_name = "insuranceAccessKey"  # Your Secrets Manager secret name
    region_name = "ap-south-1"          # Region where your secret is stored

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret
    except ClientError as e:
        print("❌ Error fetching AWS secret:", e)
        raise e


# 🧠 Prediction function using SageMaker endpoint
def predict_insurance_charge(predictor, age, bmi, children, sex_male, smoker_yes):
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    input_data = {
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [sex_male],
        "smoker_yes": [smoker_yes]
    }

    prediction = predictor.predict(input_data)
    return prediction


@app.route('/')
def index():
    return render_template('input.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form input values
        age_input = float(request.form['age'])
        bmi_input = float(request.form['bmi'])
        children_input = float(request.form['children'])
        sex_male_input = float(request.form['sex_male'])
        smoker_yes_input = float(request.form['smoker_yes'])

        # 🔐 Get AWS credentials from Secrets Manager
        aws_creds = get_aws_secrets()

        aws_access_key_id = aws_creds['AWS_ACCESS_KEY_ID']
        aws_secret_access_key = aws_creds['AWS_SECRET_ACCESS_KEY']
        region_name = "ap-south-1"
        endpoint_name = "insurance-charge-endpoint4"

        # Create boto3 session using secrets
        boto_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        # Create SageMaker session and predictor
        sagemaker_session = Session(boto_session=boto_session)
        predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)

        # Get prediction
        result = predict_insurance_charge(
            predictor,
            age=age_input,
            bmi=bmi_input,
            children=children_input,
            sex_male=sex_male_input,
            smoker_yes=smoker_yes_input
        )

        return render_template('output.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
