import numpy as np 
import pandas as pd 	
import matplotlib.pyplot as plt 
import math


def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop([df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class  GaussianNB:

	"""
		Bayes Theorem:
										Likelihood * Class prior probability
				Posterior Probability = -------------------------------------
											Predictor prior probability
				
							  			 P(x|c) * p(c)
							   P(c|x) = ------------------ 
											  P(x)

		Gaussian Naive Bayes:

							         1								
				P(x|c) = --------------------------- * exp(- (x - mean)^2 / 2*(var(x)^2)))
						   sqrt(2 * pi * var(x)^2)	
	"""

	def __init__(self):

		"""
			Attributes:
				likelihoods: Likelihood of each feature per class
				class_priors: Prior probabilities of classes  
				features: All features of dataset

		"""
		self.features = list
		self.likelihoods = {}
		self.class_priors = {}

		self.X_train = np.array
		self.y_train = np.array
		self.train_size = int
		self.num_feats = int

	def fit(self, X, y):

		self.features = list(X.columns)
		self.X_train = X
		self.y_train = y
		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]

		for feature in self.features:
			self.likelihoods[feature] = {}

			for outcome in np.unique(self.y_train):
				self.likelihoods[feature].update({outcome:{}})
				self.class_priors.update({outcome: 0})


		self._calc_class_prior()
		self._calc_likelihoods()

		# print(self.likelihoods)
		# print(self.class_priors)

	def _calc_class_prior(self):

		""" P(c) - Prior Class Probability """

		for outcome in np.unique(self.y_train):
			outcome_count = sum(self.y_train == outcome)
			self.class_priors[outcome] = outcome_count / self.train_size

	def _calc_likelihoods(self):

		""" P(x|c) - Likelihood """

		for feature in self.features:

			for outcome in np.unique(self.y_train):
				self.likelihoods[feature][outcome]['mean'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].mean()
				self.likelihoods[feature][outcome]['variance'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].var()


	def predict(self, X):

		""" Calculates Posterior probability P(c|x) """

		results = []
		X = np.array(X)

		for query in X:
			probs_outcome = {}
		
			"""
			 	Note: No Need to calculate evidence i.e P(x) since it is constant fot the given sample.
			          Therfore, it does not affect classification and can be ignored
			"""
			for outcome in np.unique(self.y_train):
				prior = self.class_priors[outcome]
				likelihood = 1
				evidence_temp = 1

				for feat, feat_val in zip(self.features, query):
					mean = self.likelihoods[feat][outcome]['mean']
					var = self.likelihoods[feat][outcome]['variance']
					likelihood *= (1/math.sqrt(2*math.pi*var)) * np.exp(-(feat_val - mean)**2 / (2*var))

				posterior_numerator = (likelihood * prior)
				probs_outcome[outcome] = posterior_numerator

		
			result = max(probs_outcome, key = lambda x: probs_outcome[x])
			results.append(result)

		return np.array(results)

			

if __name__ == "__main__":

	#Weather Dataset
	print("\nIris Dataset:")

	df = pd.read_csv("../Data/iris.csv")
	#print(df)

	#Split fearures and target
	X,y  = pre_processing(df)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

	#print(X_train, y_train)
	gnb_clf = GaussianNB()
	gnb_clf.fit(X_train, y_train)
	#print(X_train, y_train)

	print("Train Accuracy: {}".format(accuracy_score(y_train, gnb_clf.predict(X_train))))
	print("Test Accuracy: {}".format(accuracy_score(y_test, gnb_clf.predict(X_test))))
	
	#Query 1:
	query = np.array([[5.7, 2.9, 4.2, 1.3]])
	print("Query 1:- {} ---> {}".format(query, gnb_clf.predict(query)))


	#############################################################################################################

	#Gender Classification Dataset
	print("\nGender Dataset:")

	df = pd.read_csv("../Data/gender.csv")
	#print(df)

	#Split fearures and target
	X,y  = df.drop([df.columns[0]], axis = 1), df[df.columns[0]]

	X_train, y_train = X, y

	gnb_clf = GaussianNB()
	gnb_clf.fit(X_train, y_train)

	print("Train Accuracy: {}".format(accuracy_score(y_train, gnb_clf.predict(X_train))))
	
	#Query 1:
	query = np.array([[6, 130, 8]])
	print("Query 1:- {} ---> {}".format(query, gnb_clf.predict(query)))

	#Query 2:
	query = np.array([[5, 80, 6]])
	print("Query 2:- {} ---> {}".format(query, gnb_clf.predict(query)))

	#Query 3:
	query = np.array([[7, 140, 14]])
	print("Query 3:- {} ---> {}".format(query, gnb_clf.predict(query)))