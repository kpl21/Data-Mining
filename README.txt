Kevin Lu

The purpose of this project was to use the agaricuslepiota.txt data from the
UCI Machine Learning Database to predict whether or not the mushroom is either edible or
poisonous. I first used one-hot encoding to make all the categorical values of the 23 features
into binary 0 or 1 values. The formulas.py uses all the formulas needed for the models.py. 
Model.py contains the main neural network: forward and backpropagation. Since I used Python 3.9 
I needed to change the class cfile to fit the syntax of Python 3.9. In the proj_test.py
I implemented the neural network and tested to see how good the neural network model worked.
I changed the raw_input() to input() to fit the Python 3.9 syntax. There also appeared to have
been a typo in the y = Layer(stuff) part of the code and instead of 3 it should be 1 cause
we only want 1 output. I also changed the delimiting part where it parsed a comma
but since I had no commas and only spaces in my preprocessing text I changed it from
',' to ' '. I also included my source files including my training, validating, testing
data which I split in a 60%/20%/20%.

Below shows a partial iteration of the model from one of the runs:

Current iteration: 12393
Current error: 0.0001344372838147953

Current iteration: 12394
Current error: 0.00013310528515469275

Current iteration: 12395
Current error: 0.0001318937755615477

Current iteration: 12396
Current error: 0.00013059276979509506

Current iteration: 12397
Current error: 0.0006176362798582022

Current iteration: 12398
Current error: 0.00013119461007644133

Current iteration: 12399
Current error: 0.0006137616615826064

Data has converged at the 12400th run.
Neural network is done training! Hit enter to validation processing.
Error percentage on training set: 0.05062

Current iteration: 1620
Current Error: 0.00013353511923590243

Current iteration: 1621
Current Error: 0.00013347638250908636

Current iteration: 1622
Current Error: 0.00013355792347646462

Current iteration: 1623
Current Error: 0.00013456942728387692

Current iteration: 1624
Current Error: 0.000133527435217496

Current iteration: 1625
Current Error: 0.0005974080383907203

Testing done! Check out the generated output files ('testing_err.txt' and 'training_err.txt')
Error percentage on testing set: 0.02214022140221402
