import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invwishart as invw

from sds.utils import random_rotation
from sds.utils import linear_regression

from sds.stats import multivariate_normal_logpdf


class GaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1.e-16):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.mu = npr.randn(self.nb_states, self.dm_obs)
        self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

    @property
    def params(self):
        return self.mu, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_cov = value

    def mean(self, z, x=None, u=None):
        return self.mu[z, :]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z, x=None, u=None, stoch=True):
        if stoch:
            return mvn(mean=self.mean(z), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z)

    def initialize(self, x, u, **kwargs):
        kmeans = kwargs.get('kmeans', True)
        if kmeans:
            from sklearn.cluster import KMeans
            _obs = np.concatenate(x)
            km = KMeans(self.nb_states).fit(_obs)

            self.mu = km.cluster_centers_
            self.cov = np.array([np.cov(_obs[km.labels_ == k].T)
                             for k in range(self.nb_states)])
        else:
            _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
            for k in range(self.nb_states):
                self.mu[k, :] = np.mean(np.vstack([_x[0, :] for _x in x]), axis=0)
                _cov[k, ...] = np.cov(np.vstack([_x[0, :] for _x in x]), rowvar=False)
            self.cov = _cov

    def permute(self, perm):
        self.mu = self.mu[perm]
        self._sqrt_cov = self._sqrt_cov[perm]

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x in x:
            _loglik = np.column_stack([multivariate_normal_logpdf(_x, self.mean(k), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        _J = np.zeros((self.nb_states, self.dm_obs))
        _h = np.zeros((self.nb_states, self.dm_obs))
        for _x, _w in zip(x, gamma):
            _J += np.sum(_w[:, :, None], axis=0)
            _h += np.sum(_w[:, :, None] * _x[:, None, :], axis=0)

        self.mu = _h / _J

        sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        weight = np.zeros((self.nb_states, ))
        for _x, _w in zip(x, gamma):
            resid = _x[:, None, :] - self.mu
            sqerr += np.sum(_w[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(_w, axis=0)

        self.cov = sqerr / weight[:, None, None]

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            mean.append(_gamma.dot(self.mu))
        return mean

class LinearGaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1.e-16):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.K = npr.randn(self.nb_states, self.dm_act, self.dm_obs)
        self.kff = npr.randn(self.nb_states, self.dm_act)

        self._sqrt_cov = npr.randn(self.nb_states, self.dm_act, self.dm_act)

    @property
    def params(self):
        return self.K, self.kff, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.K, self.kff, self._sqrt_cov = value

    def mean(self, z, x):
        return np.einsum('kh,...h->...k', self.K[z, ...], x) + self.kff[z, ...]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_act))

    def sample(self, z, x, stoch=True):
        if stoch:
            return mvn(mean=self.mean(z, x), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z, x)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_u.shape[0] for _u in u]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(u)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [_x[t, :] for t, _x in zip(ts, x)]
            ys = [_u[t, :] for t, _u in zip(ts, u)]

            coef_, intercept_, sigma = linear_regression(xs, ys)
            self.K[k, ...] = coef_[:, :self.dm_act]
            self.kff[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.K = self.K[perm, ...]
        self.kff = self.kff[perm, ...]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            for k in range(self.nb_states):
                coef_ = np.column_stack((self.K[k, ...], self.kff[k, ...])).flatten()
                lp += mvn(mean=self.prior['mu0'] * np.zeros((coef_.shape[0], )),
                          cov=self.prior['sigma0'] * np.eye(coef_.shape[0])).logpdf(coef_)\
                      + invw(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_act)).logpdf(self.cov[k, ...])
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_u,  self.mean(k, x=_x), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
            ys.append(_u)
            ws.append(_w)

        _cov = np.zeros((self.nb_states, self.dm_act, self.dm_act))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False,
                                             **self.prior)
            self.K[k, ...] = coef_[:, :self.dm_obs]
            self.kff[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.K[k] = self.K[i] + 0.01 * npr.randn(*self.K[i].shape)
        #         self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    # def mstep(self, gamma, x, u, **kwargs):
    #     xs, ys, ws = [], [], []
    #     for _x, _u, _w in zip(x, u, gamma):
    #         xs.append(np.hstack((_x, np.ones((_x.shape[0], 1)))))
    #         ys.append(_u)
    #         ws.append(_w)
    #
    #     _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs), self.reg * np.ones(1)))
    #     _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
    #     _h = np.zeros((self.nb_states, self.dm_obs + 1, self.dm_act))
    #
    #     # solving p = (xT w x)^-1 xT w y
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         for k in range(self.nb_states):
    #             wx = _x * _w[:, k:k + 1]
    #             _J[k] += np.dot(wx.T, _x)
    #             _h[k] += np.dot(wx.T, _y)
    #
    #     mu = np.linalg.solve(_J, _h)
    #     self.K = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
    #     self.kff = mu[:, -1, :]
    #
    #     sqerr = np.zeros((self.nb_states, self.dm_act, self.dm_act))
    #     weight = self.reg * np.ones(self.nb_states)
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         yhat = np.matmul(_x[None, :, :], mu)
    #         resid = _y[None, :, :] - yhat
    #         sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
    #         weight += np.sum(_w, axis=0)
    #
    #     _cov = sqerr / weight[:, None, None]
    #
    #     usage = sum([_gamma.sum(0) for _gamma in gamma])
    #     unused = np.where(usage < 1)[0]
    #     used = np.where(usage > 1)[0]
    #     if len(unused) > 0:
    #         for k in unused:
    #             i = npr.choice(used)
    #             self.K[k] = self.K[i] + 0.01 * npr.randn(*self.K[i].shape)
    #             self.kff[k] = self.kff[i] + 0.01 * npr.randn(*self.kff[i].shape)
    #             _cov[k] = _cov[i]
    #
    #     self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x), self.nb_states, self.dm_act))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x)
            mean.append(np.einsum('nk,nkl->nl', _gamma, _mu))
        return mean


class AutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dm_obs, dm_act, prior, reg=1.e-16):
        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.A = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        self.B = np.zeros((self.nb_states, self.dm_obs, self.dm_act))
        self.c = np.zeros((self.nb_states, self.dm_obs))

        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dm_obs)
            self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
            self.c[k, :] = npr.randn(self.dm_obs)

        self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

    @property
    def params(self):
        return self.A, self.B, self.c, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sqrt_cov = value

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z, x, u, stoch=True):
        if stoch:
            return mvn(self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z, x, u)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(x)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_x[t, :], _u[t, :])) for t, _x, _u in zip(ts, x, u)]
            ys = [_x[t + 1, :] for t, _x in zip(ts, x)]

            coef_, intercept_, sigma = linear_regression(xs, ys)
            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:]
            self.c[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            for k in range(self.nb_states):
                coef_ = np.column_stack((self.A[k, ...], self.B[k, ...], self.c[k, ...])).flatten()
                lp += mvn(mean=self.prior['mu0'] * np.zeros((coef_.shape[0], )),
                          cov=self.prior['sigma0'] * np.eye(coef_.shape[0])).logpdf(coef_)\
                      + invw(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_obs)).logpdf(self.cov[k, ...])
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_x[1:, :], self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act]), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False,
                                             **self.prior)
            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:self.dm_obs + self.dm_act]
            self.c[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
        #         self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
        #         self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    # def mstep(self, gamma, x, u, **kwargs):
    #     xs, ys, ws = [], [], []
    #     for _x, _u, _w in zip(x, u, gamma):
    #         xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
    #         ys.append(_x[1:, :])
    #         ws.append(_w[1:, :])
    #
    #     _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs),
    #                               self.reg * np.ones(self.dm_act),
    #                               self.reg * np.ones(1)))
    #     _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
    #     _h = np.zeros((self.nb_states, self.dm_obs + self.dm_act + 1, self.dm_obs))
    #
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         for k in range(self.nb_states):
    #             wx = _x * _w[:, k:k + 1]
    #             _J[k] += np.dot(wx.T, _x)
    #             _h[k] += np.dot(wx.T, _y)
    #
    #     mu = np.linalg.solve(_J, _h)
    #     self.A = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
    #     self.B = np.swapaxes(mu[:, self.dm_obs:self.dm_obs + self.dm_act, :], 1, 2)
    #     self.c = mu[:, -1, :]
    #
    #     sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
    #     weight = self.reg * np.ones(self.nb_states)
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         yhat = np.matmul(_x[None, :, :], mu)
    #         resid = _y[None, :, :] - yhat
    #         sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
    #         weight += np.sum(_w, axis=0)
    #
    #     _cov = sqerr / weight[:, None, None]
    #
    #     usage = sum([_gamma.sum(0) for _gamma in gamma])
    #     unused = np.where(usage < 1)[0]
    #     used = np.where(usage > 1)[0]
    #     if len(unused) > 0:
    #         for k in unused:
    #             i = npr.choice(used)
    #             self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
    #             self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
    #             self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
    #             _cov[k] = _cov[i]
    #
    #     self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x) - 1, self.nb_states, self.dm_obs))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
            mean.append(np.einsum('nk,nkl->nl', _gamma[1:, ...], _mu))
        return mean


class RotAutoRegressiveGaussianObservation2:

    def __init__(self, nb_states, dm_obs, n_rot, prior, dm_act=0, reg=1e-8):
        self.nb_states = nb_states
        self.nb_lds = int(nb_states/n_rot)
        self.dm_obs = dm_obs
        self.dm_act = dm_act
        self.reg = reg
        self.n_rot = n_rot

        self.prior = prior


        self.rot_lds = np.zeros([self.nb_states,2])
        z = 0
        for i in range(self.nb_lds):
            for j in range(self.n_rot):
                self.rot_lds[z,0] = int(i)
                self.rot_lds[z,1] = int(j)
                z+=1

        self.A = np.zeros((self.nb_lds, self.dm_obs, self.dm_obs))
        self.B = np.zeros((self.nb_lds, self.dm_obs, self.dm_act))
        self.c = np.zeros((self.nb_lds, self.dm_obs))

        for k in range(self.nb_lds):
            self.A[k, ...] = .95 * random_rotation(self.dm_obs)
            self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
            self.c[k, :]   = npr.randn(self.dm_obs)

        self.T = np.zeros((self.n_rot, self.dm_obs, self.dm_obs))
        self.T_inv = np.zeros((self.n_rot, self.dm_obs, self.dm_obs))

        for k in range(self.n_rot):
            #self.T[k, ...] = .95 * random_rotation(self.dm_obs)
            self.T[k, ...] = np.array([[1,0],[0,1]])
            #self.T[k, ...] = np.array([[0, -1], [1, 0]])
            self.T_inv[k, ...] = np.linalg.inv(self.T[k, ...])

        self._sqrt_cov = npr.randn(self.nb_lds, self.dm_obs, self.dm_obs)

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', False)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(x)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ## Select the transformation
            si = int(self.rot_lds[k,0])
            sj = int(self.rot_lds[k,1])
            T = self.T[sj, ...]

            ts = [np.where(z == k)[0] for z in zs]

            xs = []
            ys = []
            for i in range(len(ts)):
                _x =x[i][ts[i],:]
                _x = np.dot(T,_x.T).T
                _y =x[i][ts[i]+1,:]
                _y = np.dot(T,_y.T).T

                xs.append(_x)
                ys.append(_y)

            ## THIS SHOULD NOT BE LIKE THIS , DUE TO IF SEVERAL TRANSFORMATIONS NOT WORK
            coef_, intercept_, sigma = linear_regression(xs, ys)
            self.A[si, ...] = coef_[:, :self.dm_obs]
            #self.B[k, ...] = coef_[:, self.dm_obs:]
            self.c[si, :] = intercept_
            _cov[si, ...] = sigma

            self.cov = _cov

        self.covt = np.zeros([self.nb_states,self.dm_obs,self.dm_obs])
        for k in range(self.nb_states):
            i = int(self.rot_lds[k, 0])
            j = int(self.rot_lds[k, 1])
            T_inv = self.T_inv[j, ...]
            self.covt[k, ...] = np.dot(T_inv,self.cov[i, ...])

    # vectorized in x and u only
    def mean(self, z, x, u):
        A, c, T, T_inv = self.get_rot_lds(z)
        x = np.einsum('kh,...h->...k', T, x)
        xt = np.einsum('kh,...h->...k', A, x) + c
        return np.einsum('kh,...h->...k', T_inv, xt)
        #return xt

    def get_rot_lds(self,z):
        zi = int(self.rot_lds[z,0])
        zj = int(self.rot_lds[z,1])

        A = self.A[zi, ...]
        c = self.c[zi, ...]
        T = self.T[zj, ...]
        T_inv = self.T_inv[zj, ...]
        return A, c, T, T_inv

    # one sample at a time
    def sample(self, z, x, u, stoch=False):
        mu = self.mean(z, x, u)
        j = int(self.rot_lds[z, 1])
        T_inv = self.T_inv[j, ...]
        T = self.T[j, ...]
        mu = np.einsum('kh,...h->...k', T, mu)

        if stoch:
            Hsample = mvn(mu, cov=self.cov[z, ...]).rvs()
            #return mvn(mu, cov=self.cov[z, ...]).rvs()
        else:
            Hsample = mu
            #return mu
        return np.einsum('kh,...h->...k', T_inv, Hsample)
        #return Hsample

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            T = _x.shape[0]
            _loglik = np.zeros((T - 1, self.nb_states))
            for k in range(self.nb_states):
                _mu = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
                j = int(self.rot_lds[k, 1])
                i = int(self.rot_lds[k, 0])
                T_inv = self.T_inv[j, ...]
                T = self.T[j, ...]

                _mu = np.einsum('kh,...h->...k', T, _mu)
                _x_t = np.einsum('kh,...h->...k', T, _x[1:, :])
                #_x_t = _x[1:, :]
                _loglik[:, k] = multivariate_normal_logpdf(_x_t, _mu, self.cov[i, ...])

            loglik.append(_loglik)
        return loglik

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def mstep(self, gamma, x, u):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            #xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
            xs.append(_x[:-1, :])
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs),
                                  self.reg * np.ones(self.dm_act),
                                  self.reg * np.ones(1)))
        _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_lds, 1, 1))
        _h = np.zeros((self.nb_lds, self.dm_obs + self.dm_act + 1, self.dm_obs))

        for k in range(self.nb_states):
            x_k = []
            y_k = []
            w_k = []
            Hx_k = []
            for _x, _y, _w in zip(xs, ys, ws):
                i= int(self.rot_lds[k,0])
                j= int(self.rot_lds[k,1])
                ## Transformation Matrix
                T = self.T[j, ...]
                x_ = np.einsum('kh,...h->...k', T, _x)
                y_ = np.einsum('kh,...h->...k', T, _y)
                w_ = _w[:, k]

                y_k.append(y_)
                x_k.append(x_)
                w_k.append(w_)

            coef_, intercept_, sigma = linear_regression(Xs = x_k, ys = y_k,
            weights = w_k, fit_intercept = True,
            **self.prior)

            self.A[k, ...] = coef_
            self.c[k, :] = intercept_
            self.cov[k, ...] = sigma

            # mu = np.linalg.solve(_J, _h)
            # self.A = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2) #Transpose because we computed by (wx)' x'A' = (wx)'y'
            # self.B = np.swapaxes(mu[:, self.dm_obs:self.dm_obs + self.dm_act, :], 1, 2)
            # self.c = mu[:, -1, :]

        # sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        # weight = self.reg * np.ones(self.nb_states)
        #
        # ## This should be done in order to learn the parameters with respect to number of linear functions
        # for _x, _y, _w in zip(xs, ys, ws):
        #     _yext = np.zeros([self.nb_states,_y.shape[0],_y.shape[1]])
        #     yhat = np.zeros([self.nb_states,_y.shape[0],_y.shape[1]])
        #     for k in range(self.nb_states):
        #         i = int(self.rot_lds[k, 0])
        #         j = int(self.rot_lds[k, 1])
        #         T = self.T[j, ...]
        #         T_inv = self.T_inv[j, ...]
        #         Hx = np.dot(T,_x[:,:self.dm_obs].T).T
        #
        #         _x[:, :self.dm_obs] = Hx
        #         mu_i = mu[i, ...]
        #         Hy = np.dot(_x,mu_i)
        #         ## Considero que deberia de ser mu_i * X o sino en el sample meter alreves.
        #         #Hy = np.einsum('kh,...h->...k', self.A[i, ...], Hx) + self.c[i, ...]
        #
        #
        #         yhat[k, ...] = Hy
        #         _yext[k, ...] = np.dot(T, _y.T).T
        #     #yhat = np.matmul(_x[None, :, :], mu)
        #     resid = _yext - yhat
        #     sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
        #     weight += np.sum(_w, axis=0)
        #
        # _cov = sqerr / weight[:, None, None]
        #
        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
        #         self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
        #         self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
        #         _cov[k] = _cov[i]
        #
        # self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x) - 1, self.nb_states, self.dm_obs))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
            mean.append(np.einsum('nk,nkl->nl', _gamma[1:, ...], _mu))
        return mean


class RotAutoRegressiveGaussianObservation:

    def __init__(self, nb_states, dm_obs, n_rot, prior, dm_act=0, reg=1.e-16):
        self.nb_states = nb_states
        self.nb_lds = int(nb_states / n_rot)
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.prior = prior
        self.reg = reg

        self.n_rot = n_rot
        self.rot_lds = np.zeros([self.nb_states, 2])


        self.A = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        self.B = np.zeros((self.nb_states, self.dm_obs, self.dm_act))
        self.c = np.zeros((self.nb_states, self.dm_obs))

        for k in range(self.nb_states):
            self.A[k, ...] = .95 * random_rotation(self.dm_obs)
            self.B[k, ...] = npr.randn(self.dm_obs, self.dm_act)
            self.c[k, :] = npr.randn(self.dm_obs)

        self._sqrt_cov = npr.randn(self.nb_states, self.dm_obs, self.dm_obs)

        self.T = np.zeros((self.n_rot, self.dm_obs, self.dm_obs))
        self.T_inv = np.zeros((self.n_rot, self.dm_obs, self.dm_obs))
        for k in range(self.n_rot):
            # self.T[k, ...] = .95 * random_rotation(self.dm_obs)
            self.T[k, ...] = np.array([[1, 0], [0, 1]])
            # self.T[k, ...] = np.array([[0, -1], [1, 0]])
            self.T_inv[k, ...] = np.linalg.inv(self.T[k, ...])

    @property
    def params(self):
        return self.A, self.B, self.c, self._sqrt_cov

    @params.setter
    def params(self, value):
        self.A, self.B, self.c, self._sqrt_cov = value

    def mean(self, z, x, u):
        return np.einsum('kh,...h->...k', self.A[z, ...], x) +\
               np.einsum('kh,...h->...k', self.B[z, ...], u) + self.c[z, :]

    @property
    def cov(self):
        return np.matmul(self._sqrt_cov, np.swapaxes(self._sqrt_cov, -1, -2))

    @cov.setter
    def cov(self, value):
        self._sqrt_cov = np.linalg.cholesky(value + self.reg * np.eye(self.dm_obs))

    def sample(self, z, x, u, stoch=True):
        if stoch:
            return mvn(self.mean(z, x, u), cov=self.cov[z, ...]).rvs()
        else:
            return self.mean(z, x, u)

    def initialize(self, x, u, **kwargs):
        localize = kwargs.get('localize', True)

        Ts = [_x.shape[0] for _x in x]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states, random_state=1)
            km.fit((np.vstack(x)))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_x[t, :], _u[t, :])) for t, _x, _u in zip(ts, x, u)]
            ys = [_x[t + 1, :] for t, _x in zip(ts, x)]

            coef_, intercept_, sigma = linear_regression(xs, ys)
            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:]
            self.c[k, :] = intercept_
            _cov[k, ...] = sigma

        self.cov = _cov

    def permute(self, perm):
        self.A = self.A[perm, ...]
        self.B = self.B[perm, ...]
        self.c = self.c[perm, :]
        self._sqrt_cov = self._sqrt_cov[perm, ...]

    def log_prior(self):
        lp = 0.
        if self.prior:
            for k in range(self.nb_states):
                coef_ = np.column_stack((self.A[k, ...], self.B[k, ...], self.c[k, ...])).flatten()
                lp += mvn(mean=self.prior['mu0'] * np.zeros((coef_.shape[0], )),
                          cov=self.prior['sigma0'] * np.eye(coef_.shape[0])).logpdf(coef_)\
                      + invw(self.prior['nu0'], self.prior['psi0'] * np.eye(self.dm_obs)).logpdf(self.cov[k, ...])
        return lp

    def log_likelihood(self, x, u):
        loglik = []
        for _x, _u in zip(x, u):
            _loglik = np.column_stack([multivariate_normal_logpdf(_x[1:, :], self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act]), self.cov[k])
                                       for k in range(self.nb_states)])
            loglik.append(_loglik)
        return loglik

    def mstep(self, gamma, x, u, **kwargs):
        xs, ys, ws = [], [], []
        for _x, _u, _w in zip(x, u, gamma):
            xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
            ys.append(_x[1:, :])
            ws.append(_w[1:, :])

        _cov = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            coef_, sigma = linear_regression(Xs=np.vstack(xs), ys=np.vstack(ys),
                                             weights=np.vstack(ws)[:, k], fit_intercept=False,
                                             **self.prior)
            self.A[k, ...] = coef_[:, :self.dm_obs]
            self.B[k, ...] = coef_[:, self.dm_obs:self.dm_obs + self.dm_act]
            self.c[k, ...] = coef_[:, -1]
            _cov[k, ...] = sigma

        # usage = sum([_gamma.sum(0) for _gamma in gamma])
        # unused = np.where(usage < 1)[0]
        # used = np.where(usage > 1)[0]
        # if len(unused) > 0:
        #     for k in unused:
        #         i = npr.choice(used)
        #         self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
        #         self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
        #         self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
        #         _cov[k] = _cov[i]

        self.cov = _cov

    # def mstep(self, gamma, x, u, **kwargs):
    #     xs, ys, ws = [], [], []
    #     for _x, _u, _w in zip(x, u, gamma):
    #         xs.append(np.hstack((_x[:-1, :], _u[:-1, :self.dm_act], np.ones((_x.shape[0] - 1, 1)))))
    #         ys.append(_x[1:, :])
    #         ws.append(_w[1:, :])
    #
    #     _J_diag = np.concatenate((self.reg * np.ones(self.dm_obs),
    #                               self.reg * np.ones(self.dm_act),
    #                               self.reg * np.ones(1)))
    #     _J = np.tile(np.diag(_J_diag)[None, :, :], (self.nb_states, 1, 1))
    #     _h = np.zeros((self.nb_states, self.dm_obs + self.dm_act + 1, self.dm_obs))
    #
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         for k in range(self.nb_states):
    #             wx = _x * _w[:, k:k + 1]
    #             _J[k] += np.dot(wx.T, _x)
    #             _h[k] += np.dot(wx.T, _y)
    #
    #     mu = np.linalg.solve(_J, _h)
    #     self.A = np.swapaxes(mu[:, :self.dm_obs, :], 1, 2)
    #     self.B = np.swapaxes(mu[:, self.dm_obs:self.dm_obs + self.dm_act, :], 1, 2)
    #     self.c = mu[:, -1, :]
    #
    #     sqerr = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
    #     weight = self.reg * np.ones(self.nb_states)
    #     for _x, _y, _w in zip(xs, ys, ws):
    #         yhat = np.matmul(_x[None, :, :], mu)
    #         resid = _y[None, :, :] - yhat
    #         sqerr += np.einsum('tk,kti,ktj->kij', _w, resid, resid)
    #         weight += np.sum(_w, axis=0)
    #
    #     _cov = sqerr / weight[:, None, None]
    #
    #     usage = sum([_gamma.sum(0) for _gamma in gamma])
    #     unused = np.where(usage < 1)[0]
    #     used = np.where(usage > 1)[0]
    #     if len(unused) > 0:
    #         for k in unused:
    #             i = npr.choice(used)
    #             self.A[k] = self.A[i] + 0.01 * npr.randn(*self.A[i].shape)
    #             self.B[k] = self.B[i] + 0.01 * npr.randn(*self.B[i].shape)
    #             self.c[k] = self.c[i] + 0.01 * npr.randn(*self.c[i].shape)
    #             _cov[k] = _cov[i]
    #
    #     self.cov = _cov

    def smooth(self, gamma, x, u):
        mean = []
        for _x, _u, _gamma in zip(x, u, gamma):
            _mu = np.zeros((len(_x) - 1, self.nb_states, self.dm_obs))
            for k in range(self.nb_states):
                _mu[:, k, :] = self.mean(k, _x[:-1, :], _u[:-1, :self.dm_act])
            mean.append(np.einsum('nk,nkl->nl', _gamma[1:, ...], _mu))
        return mean

