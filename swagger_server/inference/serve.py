from swagger_server.inference.thalas.thalas_detector import ThalasDetector
from swagger_server.inference.pipeline import create_pipeline_parameters


p_thalas = ThalasDetector()
p_parameters = create_pipeline_parameters()


def get_model_thalas():
    global p_thalas
    return p_thalas


def get_model_parameters():
    global p_parameters
    return p_parameters