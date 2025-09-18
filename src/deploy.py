import os
import logging
import sagemaker
from sagemaker.pytorch import PyTorchModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_S3_PATH = os.getenv("MODEL_S3_PATH")  # e.g., s3://bucket/path/to/model.tar.gz
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "eth-price-prediction-endpoint")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")  # optional; if not set, try get_execution_role()
print(MODEL_S3_PATH, "WHY NO SEE THIS")
if not MODEL_S3_PATH:
    raise ValueError("Set MODEL_S3_PATH env var to your model artifact S3 URI.")

try:
    role = ROLE_ARN or sagemaker.get_execution_role()
except Exception:
    raise RuntimeError("Could not resolve SageMaker execution role. Set SAGEMAKER_ROLE_ARN.")

pytorch_model = PyTorchModel(
    model_data=MODEL_S3_PATH,
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
)

log.info(f"🚀 Deploying to endpoint: {ENDPOINT_NAME}")
predictor = pytorch_model.deploy(
    initial_instance_count=int(os.getenv("ENDPOINT_INSTANCES", "1")),
    instance_type=os.getenv("ENDPOINT_INSTANCE_TYPE", "ml.t2.medium"),
    endpoint_name=ENDPOINT_NAME,
)
log.info(f"✅ Deployment complete. Endpoint: {predictor.endpoint_name}")