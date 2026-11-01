from flask import (
    Blueprint, jsonify, request
)
from pydantic import ValidationError

from app.services.AlgorithmService import AlgorithmService 
from ..DTOs.InputAlgorithmDTO import RecommendRequestDTO

bp = Blueprint('algorithm', __name__, url_prefix='/algorithm')


@bp.route('/recomend', methods=['POST'])
def recommend():
    try:
        dto = RecommendRequestDTO(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    recommendation = AlgorithmService.Recommend(dto) 
    
    return jsonify({"recomendações": recommendation})