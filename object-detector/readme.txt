for demo:
python .\test-classifier.py -i ../data/dataset/INRIAPerson/Test/pos/crop001604.png --visualize

# Extract the features
pos_path = "../data/dataset/INRIAPerson/Train/person"
neg_path = "../data/dataset/INRIAPerson/Train/noperson"
python extract-features.py -p ../data/dataset/INRIAPerson/Train/person -n ../data/dataset/INRIAPerson/Train/noperson

python train-classifier.py -p ../data/features/pos -n ../data/features/neg