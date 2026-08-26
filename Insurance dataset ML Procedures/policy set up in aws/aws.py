import boto3
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def predict_insurance_charge(predictor, age, bmi, children, sex_male, smoker_yes):
    """
    Sends a prediction request to the SageMaker endpoint.
    Returns: Predicted insurance charge
    """
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

# ----------- YOUR AWS CREDENTIALS ----------
aws_access_key_id = ""
aws_secret_access_key = ""
region_name = "us-east-1"
endpoint_name = "insurance-charge-serverless-endpoint"
# -------------------------------------------
# Create a boto3 session using credentials
boto_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Create a SageMaker session from the boto3 session
sagemaker_session = Session(boto_session=boto_session)

# Create Predictor object
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session
)

# Predict
result = predict_insurance_charge(
    predictor,
    age=35,
    bmi=28.4,
    children=2,
    sex_male=1,
    smoker_yes=0
)

print("✅ Predicted Insurance Charge:", result)
