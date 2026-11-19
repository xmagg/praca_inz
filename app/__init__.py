from flask import Flask


def create_app(config=None, *, prediction_service=None):
    app = Flask(__name__)

    if config is not None:
        app.config.update(config)

    from app.prediction_service import PredictionService
    from app.routes import PredictionController, main_blueprint
    from machine_learning.model_utils import HeartDiseaseModel

    service = (
        prediction_service
        if prediction_service is not None
        else PredictionService(HeartDiseaseModel())
    )
    app.extensions["prediction_service"] = service
    app.extensions["prediction_controller"] = PredictionController(service)
    app.register_blueprint(main_blueprint)

    return app
