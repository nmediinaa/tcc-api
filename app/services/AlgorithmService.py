from app.DTOs.InputAlgorithmDTO import RecommendRequestDTO 
from app.Algorithm.Recomendation import kmknn_v3


class AlgorithmService:

    def Recommend(dto: RecommendRequestDTO):
        input = dto.answers

        modelo = kmknn_v3()

        modelo.offline()

        return modelo.online(input, 1, verbose=True)

        