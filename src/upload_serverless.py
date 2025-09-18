# ------------------------------
# Step 1 & 2: Imports and Setup
# ------------------------------
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

# Setup SageMaker session and role
session = sagemaker.Session()
role = get_execution_role()

# ------------------------------
# Step 3: Define Your Model
# ------------------------------
model = Model(
    model_data='s3://sagemaker-us-east-1-469090608362/eth-price-prediction-data-train-2025-08-17-20-16-27-317/output/model.tar.gz',
    image_uri='469090608362.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:0.24-1-cpu-py3',
   role=role
)

# ------------------------------
# ✅ Step 4: Deploy to Serverless
# ------------------------------
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,   # 1024–6144 MB
    max_concurrency=10        # Requests handled concurrently
)

predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='my-serverless-endpoint'
)

predictor.delete_endpoint()
