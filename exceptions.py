"""
Domain-specific exception classes for the London Underground Delay Prediction system.

Using custom exceptions improves:
 - Clarity: callers can catch exactly the error they care about.
 - Safety: broad `except ValueError` blocks won't accidentally swallow
   unrelated standard-library errors.
 - API layer control: FastAPI handlers can map each domain error to
   the right HTTP status code without matching on error message strings.
"""


class SchemaValidationError(ValueError):
    """Raised when a DataFrame fails schema or value-range validation.

    Inherits from ValueError so existing code that catches ValueError
    continues to work without modification, while callers that need
    finer control can catch SchemaValidationError specifically.

    Example::

        try:
            validate_schema(df, config)
        except SchemaValidationError as exc:
            logger.error("Bad input data: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    """


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before a model has been loaded.

    Inherits from RuntimeError to signal a programming/operational error
    (the model file is missing or the server was not initialised) rather
    than a bad request from the caller.

    Example::

        if self.model is None:
            raise ModelNotLoadedError("Model not loaded — run train.py first")
    """


class InvalidLineError(ValueError):
    """Raised when an unknown tube line name is supplied.

    Keeps line-name validation errors distinct from generic value
    errors so the API can return an informative 422 response.
    """


class PredictionError(RuntimeError):
    """Raised when the model produces an unexpected output during inference.

    Wraps lower-level exceptions (e.g. shape mismatches, NaN outputs)
    so that the API layer can catch a single well-defined error type
    and return a 500 Internal Server Error without exposing internals.
    """
