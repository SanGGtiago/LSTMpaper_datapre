import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class MF_b():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.R_ = None
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((i, rmse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; rmse = %.4f; " % (i+1, rmse))
        
        return training_process

    def check_dif(self, R1):
        
        check = self.R
        
        self.R = R1

        for i in range(len(check)):
            for j in range(len(check[0])):
                if R1[i][j] == check[i][j]:
                    R1[i][j] = 0
        return R1

    def singletrain(self, R1, new_iter):

        R1 = self.check_dif(R1)
        
        self.samples = [
            (i, j, R1[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if R1[i, j] > 0
        ]

        training_process = []
        for i in range(new_iter):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((i, rmse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, rmse))
        
        return training_process

    def rmse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error/len(xs))

    def rating_error(self, new_mf):
        #########rating#########
        new_mf = self.check_dif(new_mf)
        xs, ys = new_mf.nonzero()
        predicted = self.full_matrix()
        rmse_error = 0
        mae_error = 0
        hit = 0
        check = 0
        for x, y in zip(xs, ys):
            rmse_error += pow(new_mf[x, y] - predicted[x, y], 2)
            mae_error += abs(new_mf[x, y] - predicted[x, y])
            if abs(new_mf[x, y] - predicted[x, y]) <0.3:
                hit += 1
            check += 1

        print(f"rating:{len(xs)}")
        print(hit/len(xs))
        print(hit)
        print(check)
        return np.sqrt(rmse_error/len(xs)), mae_error/len(xs)

    def rate_mae(self, R1):
        """
        A function to compute the total mean square error
        """
        xs, ys = R1.nonzero()
        predicted = self.full_matrix()
        error = 0
        hit = 0
        for x, y in zip(xs, ys):
            error += abs(R1[x, y] - predicted[x, y])
            if abs(R1[x, y] - predicted[x, y]) < 0.3:
                hit += 1
        print(hit/len(xs))
        print(f"rate:{len(xs)}")
        return error/len(xs)

    def rate_rmse(self, R1):
        """
        A function to compute the total mean square error
        """
        xs, ys = R1.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(R1[x, y] - predicted[x, y], 2)

        return np.sqrt(error/len(xs))

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)