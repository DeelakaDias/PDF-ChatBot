import torch
from InstructorEmbedding import INSTRUCTOR

print("Torch version:", torch.__version__)

try:
    model = INSTRUCTOR('hkunlp/instructor-x1')
    print("InstructorEmbedding loaded successfully!")
except Exception as e:
    print("Error loading InstructorEmbedding:", e)