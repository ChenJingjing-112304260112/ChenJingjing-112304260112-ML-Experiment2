print("Testing Python environment...")

try:
    import pandas
    print("+ pandas installed")
except ImportError:
    print("- pandas not installed")

try:
    import numpy
    print("+ numpy installed")
except ImportError:
    print("- numpy not installed")

try:
    import nltk
    print("+ nltk installed")
except ImportError:
    print("- nltk not installed")

try:
    import gensim
    print("+ gensim installed")
except ImportError:
    print("- gensim not installed")

try:
    import sklearn
    print("+ scikit-learn installed")
except ImportError:
    print("- scikit-learn not installed")

print("Environment test completed.")