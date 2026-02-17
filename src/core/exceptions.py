class PumpFaultRiskException(Exception):
    """Base class for exceptions in the Pump Fault Risk service."""
    pass

class ModelNotFoundException(PumpFaultRiskException):
    """Exception raised when the model is not found."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not found.")

class InvalidInputException(PumpFaultRiskException):
    """Exception raised for invalid input data."""
    def __init__(self, message: str):
        super().__init__(message)

class PredictionErrorException(PumpFaultRiskException):
    """Exception raised when an error occurs during prediction."""
    def __init__(self, message: str):
        super().__init__(message)