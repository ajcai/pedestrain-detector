This is a project of pedetrain detection with HOG and SVM.
Main steps are: 1.sample image patches -> 2.extract features -> 3.train a SVM model 
-> 4.collect hard examples -> 5.retrain the model -> 6.demo or test on Test dataset
#for demo:
cd object-detector
python demo.py

#train a model
cd object-detector
python main.py