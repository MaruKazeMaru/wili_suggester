import numpy as np
from numpy import ndarray
from .prob import rand_unform_cube, rand_uniform_sinplex, calc_stat_dist

class suggester:
    motion_num:int
    tr_prob:ndarray
    init_prob:ndarray
    avr_where_user:list[ndarray]
    var_where_user:list[ndarray]
    inv_var:list[ndarray]
    gauss_divs:list[float]

    burn_in:int
    skip:int
    noreject_sample_num:int
    all_sample_num:int
    sample:ndarray
    dens_sample:ndarray


    def __init__(self, motion_num:int, \
                    tr_prob:ndarray=None, init_prob:ndarray=None, \
                    avr_where_user:list[ndarray]=None, var_where_user:list[ndarray]=None, \
                    burn_in=30, skip=3, noreject_sample_num=500 \
                ):
        # ~~~ HMM ~~~
        # node
        self.motion_num = motion_num

        # transition probability
        if tr_prob is None:
            self.tr_prob = rand_uniform_sinplex(self.motion_num, num=self.motion_num).T
        else:
            self.tr_prob = tr_prob

        if init_prob is None:
            self.init_prob = calc_stat_dist(self.tr_prob)
        else:
            self.init_prob = init_prob

        # Gaussian
        if avr_where_user is None:
            v = np.array([1.0, 0.0])
            ang = 2.0 * np.pi / float(self.motion_num)
            c = np.cos(ang)
            s = np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            self.avr_where_user = []
            for i in range(self.motion_num):
                self.avr_where_user.append(v)
                v = R @ v
        else:
            self.avr_where_user = avr_where_user

        if var_where_user is None:
            self.var_where_user = []
            for i in range(self.motion_num):
                self.var_where_user.append(np.identity(2))
        else:
            self.var_where_user = var_where_user

        # cache
        dets = [np.linalg.det(Sigma) for Sigma in self.var_where_user]
        self.gauss_divs = [2.0 * np.pi * det for det in dets]
        self.inv_var = [np.linalg.inv(v) for v in self.var_where_user]

        # ~~~ MCMC ~~~
        self.burn_in = burn_in
        self.skip = skip
        self.noreject_sample_num = noreject_sample_num
        self.all_sample_num = self.burn_in + 1 + (self.noreject_sample_num - 1) * self.skip
        self.sample = rand_unform_cube(self.motion_num, num=self.all_sample_num)
        self.dens_sample = np.ones((self.all_sample_num,))


    def weight(self, miss_prob:ndarray) -> ndarray:
        L = miss_prob.reshape((self.motion_num, 1)) * self.tr_prob.T
        for i in range(self.motion_num):
            L[i,i] = 0.0
        K = self.tr_prob.T - L
        return L @ np.linalg.inv(np.identity(self.motion_num) - K) @ self.init_prob


    def gaussian(self, x:ndarray) -> ndarray:
        e = ndarray((self.motion_num,))
        x_s = [x - myu for myu in self.avr_where_user]
        for i in range(self.motion_num):
            e[i] = x_s[i] @ self.inv_var[i] @ x_s[i]
            e[i] = np.exp(-0.5 * e[i])
            e[i] /= self.gauss_divs[i]
        return e


    def liklyhood(self, miss_prob:ndarray, x:ndarray=None) -> float:
        return np.dot(self.weight(miss_prob), self.gaussian(x))


    def expectation(self, f, f_kwargs:dict={}) -> float | ndarray:
        p = -1.0
        sum_f = 0.0
        for i in range(self.all_sample_num):
            p_ = self.dens_sample[i]
            if (p_ >= p) or (np.random.rand() * p < p_):
                p = p_

            if (i >= self.burn_in) and ((i - self.burn_in) % self.skip == 0):
                sum_f += f(self.sample[:,i], **f_kwargs)
        return sum_f / self.noreject_sample_num


    def update(self, where_found:ndarray) -> None:
        exp_l = np.dot(self.expectation(self.weight), self.gaussian(where_found))
        for i in range(self.all_sample_num):
            self.dens_sample[i] = self.liklyhood(self.sample[:,i], x=where_found) * self.dens_sample[i]
        self.dens_sample /= exp_l


    def suggest(self) -> ndarray:
        return self.expectation(self.weight)
