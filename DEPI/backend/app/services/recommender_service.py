from app.recommenders.doctor_model import get_mock_doctor_recommendations
from app.recommenders.lifestyle_model import get_mock_lifestyle_recommendations


def get_mock_recommendations() -> dict[str, list[dict[str, str]]]:
    return {
        "lifestyle": get_mock_lifestyle_recommendations(),
        "doctors": get_mock_doctor_recommendations(),
    }
