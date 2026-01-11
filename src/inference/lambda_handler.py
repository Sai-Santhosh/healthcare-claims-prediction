"""
AWS Lambda Handler for Claims Prediction
Serverless inference endpoint for predicting paid amounts.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Try to import boto3 (available in Lambda environment)
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger.warning("boto3 not available - S3 model loading disabled")


@dataclass
class PredictionRequest:
    """Request schema for predictions."""
    amt_billed: float
    amt_deduct: float = 0.0
    amt_coins: float = 0.0
    age: int = 45
    gender_code: int = 0
    client_los: float = 0.0
    form_type: str = 'P'
    sv_stat: str = 'P'
    product_type: str = 'PPO'
    icd_category: str = 'Z'


@dataclass
class PredictionResponse:
    """Response schema for predictions."""
    predicted_amount: float
    confidence_interval: Dict[str, float]
    model_version: str
    request_id: str


class ClaimPredictor:
    """
    Claims prediction service.
    Handles model loading and inference.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_key: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Local path to model file
            s3_bucket: S3 bucket name
            s3_key: S3 object key for model
        """
        self.model = None
        self.transformer = None
        self.feature_names = None
        self.model_version = "unknown"
        
        if model_path and os.path.exists(model_path):
            self._load_local_model(model_path)
        elif s3_bucket and s3_key and S3_AVAILABLE:
            self._load_s3_model(s3_bucket, s3_key)
        else:
            logger.warning("No model loaded - predictor not ready")
    
    def _load_local_model(self, model_path: str) -> None:
        """Load model from local file."""
        logger.info(f"Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Try to load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.model_version = metadata.get('version', 'unknown')
        
        logger.info(f"Model loaded successfully (version: {self.model_version})")
    
    def _load_s3_model(self, bucket: str, key: str) -> None:
        """Load model from S3."""
        logger.info(f"Loading model from S3: s3://{bucket}/{key}")
        
        s3 = boto3.client('s3')
        
        # Download model
        response = s3.get_object(Bucket=bucket, Key=key)
        self.model = pickle.loads(response['Body'].read())
        
        # Try to load metadata
        try:
            metadata_key = key.replace('.pkl', '_metadata.json')
            response = s3.get_object(Bucket=bucket, Key=metadata_key)
            metadata = json.loads(response['Body'].read())
            self.feature_names = metadata.get('feature_names', [])
            self.model_version = metadata.get('version', 'unknown')
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        logger.info(f"Model loaded from S3 (version: {self.model_version})")
    
    def _prepare_features(self, request: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features from request for model prediction.
        
        Args:
            request: Request dictionary with claim data
        
        Returns:
            Feature array for prediction
        """
        # Extract features
        features = {
            'AMT_BILLED': request.get('amt_billed', 0),
            'AMT_DEDUCT': request.get('amt_deduct', 0),
            'AMT_COINS': request.get('amt_coins', 0),
            'Age': request.get('age', 45),
            'Gender_Code': request.get('gender_code', 0),
            'CLIENT_LOS': request.get('client_los', 0),
        }
        
        # Add log feature
        features['AMT_BILLED_log'] = np.log1p(features['AMT_BILLED'])
        
        # Create dummy variables for categorical features
        form_type = request.get('form_type', 'P')
        for ft in ['I', 'O', 'P']:
            features[f'FORM_TYPE_{ft}'] = 1 if form_type == ft else 0
        
        sv_stat = request.get('sv_stat', 'P')
        for ss in ['D', 'P', 'R']:
            features[f'SV_STAT_{ss}'] = 1 if sv_stat == ss else 0
        
        product_type = request.get('product_type', 'PPO')
        for pt in ['HMO', 'POS', 'PPO']:
            features[f'PRODUCT_TYPE_{pt}'] = 1 if product_type == pt else 0
        
        icd_cat = request.get('icd_category', 'Z')
        for cat in ['E', 'F', 'I', 'J', 'K', 'M', 'Z']:
            features[f'ICD_Category_{cat}'] = 1 if icd_cat == cat else 0
        
        # If we have feature names, ensure correct order
        if self.feature_names:
            feature_array = []
            for name in self.feature_names:
                feature_array.append(features.get(name, 0))
            return np.array([feature_array])
        else:
            return np.array([list(features.values())])
    
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a claim.
        
        Args:
            request: Request dictionary with claim data
        
        Returns:
            Prediction response dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare features
        features = self._prepare_features(request)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        # Calculate confidence interval (simplified)
        # In production, you'd use model-specific uncertainty quantification
        std_estimate = prediction * 0.15  # Assume 15% standard error
        ci_lower = max(0, prediction - 1.96 * std_estimate)
        ci_upper = prediction + 1.96 * std_estimate
        
        return {
            'predicted_amount': round(prediction, 2),
            'confidence_interval': {
                'lower': round(ci_lower, 2),
                'upper': round(ci_upper, 2)
            },
            'model_version': self.model_version
        }
    
    def predict_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple claims.
        
        Args:
            requests: List of request dictionaries
        
        Returns:
            List of prediction responses
        """
        return [self.predict(req) for req in requests]


# Global predictor instance (loaded once per Lambda container)
_predictor: Optional[ClaimPredictor] = None


def get_predictor() -> ClaimPredictor:
    """Get or initialize the global predictor."""
    global _predictor
    
    if _predictor is None:
        # Try environment variables first
        s3_bucket = os.environ.get('MODEL_S3_BUCKET')
        s3_key = os.environ.get('MODEL_S3_KEY')
        local_path = os.environ.get('MODEL_LOCAL_PATH')
        
        _predictor = ClaimPredictor(
            model_path=local_path,
            s3_bucket=s3_bucket,
            s3_key=s3_key
        )
    
    return _predictor


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event (API Gateway request)
        context: Lambda context
    
    Returns:
        API Gateway response
    """
    logger.info(f"Received event: {json.dumps(event)[:500]}")
    
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Get predictor
        predictor = get_predictor()
        
        # Handle batch or single prediction
        if isinstance(body, list):
            predictions = predictor.predict_batch(body)
        else:
            predictions = predictor.predict(body)
        
        # Get request ID
        request_id = context.aws_request_id if context else 'local'
        
        # Build response
        response_body = {
            'success': True,
            'request_id': request_id,
            'predictions': predictions if isinstance(body, list) else predictions
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': 'Invalid request',
                'message': str(e)
            })
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': 'Internal server error',
                'message': str(e)
            })
        }


# For local testing
if __name__ == '__main__':
    # Test request
    test_event = {
        'body': json.dumps({
            'amt_billed': 1500.00,
            'amt_deduct': 100.00,
            'amt_coins': 50.00,
            'age': 45,
            'gender_code': 1,
            'form_type': 'P',
            'product_type': 'PPO'
        })
    }
    
    # Mock context
    class MockContext:
        aws_request_id = 'test-123'
    
    result = handler(test_event, MockContext())
    print(json.dumps(json.loads(result['body']), indent=2))
