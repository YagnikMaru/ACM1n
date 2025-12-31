# AutoJudge: Programming Problem Difficulty Predictor

## Project Overview
AutoJudge is a machine learning system that automatically predicts the difficulty of programming problems based on their textual descriptions. The system classifies problems as Easy, Medium, or Hard and predicts a continuous difficulty score between 0 and 10.

### Key Features
- **Dual Prediction**: Both classification (Easy/Medium/Hard) and regression (0-10 score)
- **Text Analysis**: Uses only textual information (no code execution)
- **Web Interface**: User-friendly web application for predictions
- **Explainable Models**: Traditional ML models with feature importance analysis
- **Production Ready**: Complete training pipeline and deployment setup

## Dataset Used
The system uses a custom dataset of programming problems with the following structure:
```json
{
  "title": "Problem Title",
  "description": "Detailed problem description",
  "input_description": "Input format specification",
  "output_description": "Output format specification",
  "problem_class": "Easy/Medium/Hard",
  "problem_score": 0-10
}