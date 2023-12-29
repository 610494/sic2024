import random
import numpy as np

def _generate_audio(sr, len):
    return np.random.rand(sr * 5)
def generate_audio(number = 3, sr = 16_000, maxLen = 10, minLen = 3):
    resultList = []
    
    for _ in range(number):
        resultList.append(_generate_audio(sr, random.randint(minLen, maxLen)))
    return resultList