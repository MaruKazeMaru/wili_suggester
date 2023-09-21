import rclpy
from rclpy.node import Node
import numpy as np
from numpy import ndarray
from prob import rand_uniform_sinplex, calc_stat_dist, rand_unform_cube

class Suggester(Node):
    motion_num:int
    tr_prob:ndarray
    init_prob:ndarray
    avr_where_user:list[ndarray]
    var_where_user:list[ndarray]
    inv_var:list[ndarray]
    gauss_divs:list[float]

    sample_num:int
    sample:ndarray
    dens_sample:ndarray

    def __init__(self):
        super().__init__('suggester')

        self.motion_num = 8
        self.tr_prob = rand_uniform_sinplex(self.motion_num, num=self.motion_num).T
        self.init_prob = calc_stat_dist(self.tr_prob)
        self.avr_where_user = []
        self.var_where_user = []
        self.gauss_divs = []
        for i in range(self.motion_num):
            self.avr_where_user.append(np.zeros((self.motion_num,)))
            self.var_where_user.append(np.identity(self.motion_num))
        dets = [np.linalg.det(Sigma) for Sigma in self.var_where_user]
        self.gauss_divs = [2.0 * np.pi * det for det in dets]
        self.inv_var = [np.linalg.inv(v) for v in self.var_where_user]

        self.sample_num = 500
        self.sample = rand_unform_cube(self.motion_num, num=self.sample_num)
        self.dens_sample = np.ones((self.sample_num,))


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
        # theta:ndarray
        # p = 0.0
        # sample_update_num = sample_size * skip + burn_in
        # sum_f = 0.0
        # for i in range(sample_update_num):
        #     theta_ = rand_uniform(self.dim)
        #     p_ = self.prob_density(theta_)
        #     #if (p_ >= 0.9999 * p) or (np.random.rand() * p > p_):
        #     if (p_ >= p) or (np.random.rand() * p < p_):
        #         p = p_
        #         theta = theta_

        #     if (i >= burn_in) and ((i - burn_in) % skip == 0):
        #         sum_f += f(theta, **f_kwargs)

        # return sum_f / sample_size


    def update(self, where_found:ndarray) -> None:
        exp_l = self.expectation(self.liklyhood, f_kwargs={"x":where_found})
        for i in range(self.sample_num):
            self.dens_sample[i] = self.liklyhood(self.sample[:,i], x=where_found) * self.dens_sample[i]
        self.dens_sample /= exp_l



def main():
    rclpy.init()
    node = Suggester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
