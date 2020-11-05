from DistributionCreator import DistributionCreator
from Estimator import Estimator
from Question1 import Main


class Tests:

    ###############tests#################
    def test01(self):
        print("hello")
        print(Main.gen_from_seed(10, 10, 0, 1))
        print(Main.gen_from_seed(10, 10, 0, 1))

    def test02(self):  # test function part A
        dist_creator = DistributionCreator()
        arr = Main.gen_from_seed(seed=500, array_size=10, lower_lim=0, upper_lim=1)
        a = dist_creator.exponential_dist(arr, mean=120000)
        b = dist_creator.weibull_dist(arr, n=7600, m=1.2)
        c = dist_creator.normal_dist(arr, mean=42000, std=663)
        d = dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2)
        e = dist_creator.gumbel_dist(arr, mean=65000, std=370)