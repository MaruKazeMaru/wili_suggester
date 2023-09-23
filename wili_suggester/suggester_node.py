#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2023 ShinagwaKazemaru
# SPDX-License-Identifier: MIT License

import rclpy
from rclpy.node import Node
import numpy as np
from wili_msgs.msg import Where
from wili_msgs.srv import GetSuggest
from .prob import rand_uniform_sinplex, calc_stat_dist, rand_unform_cube
from .suggester import Suggester

class SuggesterNode(Node, Suggester):
    def __init__(self):
        Node.__init__(self, 'suggester')
        self.logger = self.get_logger()

        Suggester.__init__(self, 3) # 3 is for test

        # print(self.motion_num)
        # print(self.tr_prob)
        # print('[')
        # for i in range(self.motion_num):
        #     print(self.avr_where_user[i])
        # print(']')
        # print('[')
        # for i in range(self.motion_num):
        #     print(self.var_where_user[i])
        # print(']')

        self.create_service(GetSuggest, 'get_suggest', self.cb_suggest)
        self.sub_update = self.create_subscription(Where, 'where_found', self.cb_update, 10)

        self.logger.info('start')


    def cb_update(self, msg:Where):
        where_found = np.array([msg.x, msg.y])
        self.logger.info('subscribed : where_found={}'.format(where_found))
        self.update(where_found)
        self.logger.info('updated')


    def cb_suggest(self, req:GetSuggest.Request, res:GetSuggest.Response) -> GetSuggest.Response:
        res.weight = self.suggest().tolist()
        self.logger.info('{}'.format(res.weight))
        return res


def main():
    rclpy.init()
    node = SuggesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: print('')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
