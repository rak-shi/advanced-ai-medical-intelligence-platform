from app.services.gemini_service import (
    generate_medical_report
)


report = generate_medical_report(
    prediction="PNEUMONIA",
    confidence=98.27,
    normal_probability=1.73,
    pneumonia_probability=98.27
)


print("\nGenerated Medical Report")
print("=" * 60)

print(report)