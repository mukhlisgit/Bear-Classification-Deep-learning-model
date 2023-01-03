# Bear-Classification-Deep-learning-model
This is a deep learning program that calculates if a bear is a grizzly bear or a polar bear
	Hypothesis

o	There are two common species of an animal called bear. They are brown bear and polar bear. This program’s aim is to efficiently differentiate between a polar bear and a brown bear.

	Hyperparameters

For the hyperparameters selection I used the sklearn library which is otherwise known as sickit learn. In this library the hyperparameters are usually passed as arguments to the constructor of the estimator classes. Then I further selected the ‘GridSearchCV’ as an approach to consider all the possible combinations when training. The values were passed according to the unique format of the ‘GridSearchCV’, which specifies that “two grids should be explored: one with a linear kernel and C values in [1, 10, 100, 1000], and the second one with an RBF kernel, and the cross-product of C values ranging in [1, 10, 100, 1000] and gamma values in [0.001, 0.0001]”.

I selected these values to enable the ‘GridSearchCV’ to evaluate all the combinations of the parameter values and then retain the best possible combination.



	Training and testing sets

In training the model I chose ‘svc’ from the sklearn library as an input along with the hyperparameters explained above and then fit x and y values into the model training. The arrays were all in one dimension because they were initially flattened. 
The accuracy was 0.917 and this is even more than enough as it is very near to 1.

	Changing the hyperparameters

Changing the hyperparameters doesn’t make any difference which Is most probably due to the ‘GridSearchCV’s ability to combine all possible parameters and come up with the best ones as earlier discussed. However, rerunning the program alters the accuracy as new datasets are passed and the approach used by ‘GridSearchCV’ in training the models will also change, thereby changing the accuracy value.
