from classifier import classifier
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class SVC(classifier):
    def __init__(self, c=1.0, tolerance=0.0001, max_iter=10000):
        self.X = None               # X observations (nxp matrix)
        self.Y = None               # labels
        self.m = None               # max number of rows/observations in X
        self.n = None
        self.w = None
        self.alphas = None          # support vectors
        self.eCache = None          # first column is valid flag TODO: ???
        self.b = 0
        self.C = c                  # budget for amount that the margin can be violated
        self.tol = tolerance        # tolerance for stopping criteria?
        self.max_iter = max_iter

    def fit(self, x, y):
        """
        Fits the X data using the  Linear SVC Model.
        :param x: Samples
        :param y: Labels
        :return: weights
        """
        self.init_class_variables(x, y)
        return self.smo_pk()

    def predict(self, X):
        """
        Using a Linear SVC fitted model, predicts labels for X.
        :param X: Samples
        :return:
        """
        fx = calc_line(X, self.w, self.b)
        hyp = apply_along_axis(lambda x: 1 if x >= 0 else -1, 0, fx)
        return hyp

    def smo_pk(self):
        n_iter = 0
        entire = True
        alpha_pairs_changed = 0
        while (n_iter < self.max_iter) and ((alpha_pairs_changed > 0) or entire):
            alpha_pairs_changed = 0
            if entire:  # go over all
                for i in range(self.m):  # iterate over number of observations
                    alpha_pairs_changed += self.inner_l_k(i)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (n_iter, i, alpha_pairs_changed))
                n_iter += 1
            else:  # go over non-bound (railed) alphas
                non_bound_is = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in non_bound_is:
                    alpha_pairs_changed += self.inner_l_k(i)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (n_iter, i, alpha_pairs_changed))
                n_iter += 1

            if entire:
                entire = False  # toggle entire set loop
            elif alpha_pairs_changed == 0:
                entire = True
            print("iteration number: %d" % n_iter)
        return self.calc_ws()

    def init_class_variables(self, x, y):
        self.X = mat(x)
        self.Y = mat(y).transpose()
        self.m = shape(x)[0]
        self.m, n = shape(self.X)
        self.w = zeros((n, 1))
        self.alphas = mat(zeros((self.m, 1)))
        self.eCache = mat(zeros((self.m, 2)))

    def calc_ws(self):
        for i in range(self.m):
            self.w += multiply(self.alphas[i] * self.Y[i], self.X[i, :].T)
        return self.w

    def calc_ek_k(self, k):
        """
        Calculates weight for observation, k
        :param k:
        :return:
        """
        # f = w^T * x
        # f = (a * y)T * (X*(X_k)T) + b
        f = float(multiply(self.alphas, self.Y).T * (self.X * self.X[k, :].T)) + self.b
        # w = f - y
        w = f - float(self.Y[k])
        return w

    def select_j_k(self, i, e_i):         #this is the second choice -heurstic, and calcs Ej
        max_k = -1
        max_delta_e = 0
        e_j = 0
        self.eCache[i] = [1, e_i]  #set valid #choose the alpha that gives the maximum delta E
        valid_ecache_list = nonzero(self.eCache[:, 0].A)[0]
        if (len(valid_ecache_list)) > 1:
            for k in valid_ecache_list:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i:
                    continue #don't calc for i, waste of time
                e_k = self.calc_ek_k(k)
                delta_e = abs(e_i - e_k)
                if delta_e > max_delta_e:
                    max_k = k
                    max_delta_e = delta_e
                    e_j = e_k
            return max_k, e_j
        else:   #in this case (first time around) we don't have any valid eCache values
            j = select_j_rand(i, self.m)
            e_j = self.calc_ek_k(j)
        return j, e_j

    def update_ek_k(self, k): # after any alpha has changed update the new value in the cache
        e_k = self.calc_ek_k(k)
        self.eCache[k] = [1, e_k]

    def inner_l_k(self, i):

        Ei = self.calc_ek_k(i)

        if ((self.Y[i]*Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.Y[i]*Ei > self.tol) and (self.alphas[i] > 0)):

            j, Ej = self.select_j_k(i, Ei)                     # this has been changed from selectJrand

            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()

            if self.Y[i] != self.Y[j]:
                l = max(0, self.alphas[j] - self.alphas[i])
                h = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                l = max(0, self.alphas[j] + self.alphas[i] - self.C)
                h = min(self.C, self.alphas[j] + self.alphas[i])
            if l == h:
                print("L==H")
                return 0

            # eta = 2 * X_i * X_j - X_i * (X_i)T - X_j * (X_j)T
            eta = (2.0 * self.X[i, :] * self.X[j, :].T) - (self.X[i, :] * self.X[i, :].T) - (self.X[j, :] * self.X[j, :].T)
            if eta >= 0:
                print("eta>=0")
                return 0

            self.alphas[j] -= self.Y[j]*(Ei - Ej)/eta
            self.alphas[j] = clip_alpha(j, h, l)
            self.update_ek_k(j)     # added this for the Ecache
            if abs(self.alphas[j] - alpha_j_old) < 0.00001:
                print("j not moving enough"); return 0
            self.alphas[i] += self.Y[j]*self.Y[i]*(alpha_j_old - self.alphas[j])#update i by the same amount as j
            self.update_ek_k(i) # added this for the Ecache                    #the update is in the oppselftie direction
            b1 = self.b - Ei - self.Y[i]*(self.alphas[i]-alpha_i_old)*self.X[i, :]*self.X[i, :].T - self.Y[j]*(self.alphas[j]-alpha_j_old)*self.X[i,:]*self.X[j, :].T
            b2 = self.b - Ej - self.Y[i]*(self.alphas[i]-alpha_i_old)*self.X[i, :]*self.X[j, :].T - self.Y[j]*(self.alphas[j]-alpha_j_old)*self.X[j,:]*self.X[j, :].T
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2)/2.0
            return 1
        else:
            return 0


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def calc_line(X, weights, b):
    return dot(weights.T, X.T) + b


def select_j_rand(i, m):
    j=i             #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


def plot_boundary(X, Y, weights, b, filename="SVM_linearly_separable.png"):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(10, 8))

    df = pd.DataFrame(columns=["x1", "x2", "label"])
    df["x1"] = X[:, 0]
    df["x2"] = X[:, 1]
    df["label"] = Y

    labels = df["label"].unique()

    ax.set_xlim(df["x1"].min() - 1, df["x1"].max() + 1)
    ax.set_ylim(df["x2"].min() - 1, df["x2"].max() + 1)

    sns.regplot(x="x1", y="x2", data=df[df["label"] == labels[0]], color="lightblue", fit_reg=False, ci=None, ax=ax)
    sns.regplot(x="x1", y="x2", data=df[df["label"] == labels[1]], color="lightgreen", fit_reg=False, ci=None, ax=ax)

    a = -weights[0] / weights[1]
    xx = linspace(df["x1"].min()-1, df["x1"].max()+1)
    yy = multiply(a, xx) - b / weights[1]
    yy = squeeze(asarray(yy))

    sns.regplot(x=xx, y=yy, ci=None, ax=ax)


    # plt.show()
    plt.savefig(filename)
    print("Saved Plot: {}".format(filename))
