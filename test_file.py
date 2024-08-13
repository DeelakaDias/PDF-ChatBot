import torch
from InstructorEmbedding import INSTRUCTOR

print("Torch version:", torch.__version__)

# Check if the model loads correctly (example model, adjust if needed)
try:
    model = INSTRUCTOR('hkunlp/instructor-x1')
    print("InstructorEmbedding loaded successfully!")
except Exception as e:
    print("Error loading InstructorEmbedding:", e)