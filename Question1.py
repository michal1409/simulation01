from numpy import random
import chaospy as cy
import heapq
from DistributionCreator import DistributionCreator
from Estimator import Estimator


class Main:
    # Generate random numbers from uniform distribution
    @staticmethod
    def gen_from_seed(seed, array_size, lower_lim, upper_lim):
        random.seed(seed)
        ret = random.uniform(lower_lim, upper_lim, array_size)
        return ret

    @staticmethod
    def gen_from_halton(array_size, lower_lim, upper_lim):
        uniform = cy.Uniform(lower_lim, upper_lim)
        samples = uniform.sample(array_size, rule="halton")
        return samples


####################### prints for delivery #####################
def estimate_distribution_parameters(seed, array_size):
    dist_creator = DistributionCreator()
    estimator = Estimator()
    print()
    print(f"Estimate with {array_size}:")
    arr = Main.gen_from_seed(seed=seed, array_size=array_size, lower_lim=0, upper_lim=1)
    # Exponential
    print(estimator.estimate_mean(dist_creator.exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential",
                                  onlyvalue=True))
    # Weibull
    print(estimator.estimate_m_n(dist_creator.weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull",
                                 onlyvalue=True))
    # Normal
    normal_arr = dist_creator.normal_dist(arr, mean=42000, std=663)
    print(estimator.estimate_mean(normal_arr, mean=42000, dist_name="Normal", onlyvalue=True))
    print(estimator.estimate_std(normal_arr, std=663, dist_name="Normal", onlyvalue=True))
    # Log Normal
    log_normal_arr = dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2)
    print(estimator.estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal", onlyvalue=True))
    print(estimator.estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal", onlyvalue=True))
    # Gumbel
    print(estimator.estimate_mean(dist_creator.gumbel_dist(arr, mean=65000, std=370), mean=65000, dist_name="Gumbel",
                                  onlyvalue=True))
    print(estimator.estimate_std(dist_creator.gumbel_dist(arr, mean=65000, std=370), std=370, dist_name="Gumbel",
                                 onlyvalue=True))


def estimate_distribution_parameters_halton(array_size):
    dist_creator = DistributionCreator()
    estimator = Estimator()
    print()
    print(f"Estimate with {array_size}:")
    arr = Main.gen_from_halton(array_size=array_size, lower_lim=0, upper_lim=1)
    # Exponential
    print(estimator.estimate_mean(dist_creator.exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential", onlyvalue=True))
    # Weibull
    print(estimator.estimate_m_n(dist_creator.weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull", onlyvalue=True))
    # Normal
    normal_arr = dist_creator.normal_dist(arr, mean=42000, std=663)
    print(estimator.estimate_mean(normal_arr, mean=42000, dist_name="Normal", onlyvalue=True))
    print(estimator.estimate_std(normal_arr, std=663, dist_name="Normal", onlyvalue=True))
    # Log Normal
    log_normal_arr = dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2)
    print(estimator.estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal", onlyvalue=True))
    print(estimator.estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal", onlyvalue=True))
    # Gumbel
    print(estimator.estimate_mean(dist_creator.gumbel_dist(arr, mean=65000, std=370), mean=65000, dist_name="Gumbel", onlyvalue=True))
    print(estimator.estimate_std(dist_creator.gumbel_dist(arr, mean=65000, std=370), std=370, dist_name="Gumbel", onlyvalue=True))


############### answers #################

# A
# generate numbers from seed
Main.gen_from_seed(seed=1, array_size=10, lower_lim=0, upper_lim=1)

# B
# Estimate with 500 numbers - good estimation
estimate_distribution_parameters(1, 500)

# C
# Estimate with 500 numbers and same seed - will be the same
estimate_distribution_parameters(1, 500)
# Estimate with different seed - the estimations is still good, the outcome is not exactly the same numbers
estimate_distribution_parameters(5, 500)
# Estimate with 1000 numbers and second seed - is it better? is it slow conversions?(1000 better then 500)
estimate_distribution_parameters(10, 1000)

# D
def D():
    est01 = [], est02 = [], est03 = [], est04 = [], est05 = [], est06 = [], est07 = [], est08 = [], est09 = [], est10 = []
    dist_creator = DistributionCreator()
    estimator = Estimator()
    for i in range(100):
        new_seed = random.randint()
        arr = Main.gen_from_seed(seed=new_seed, array_size=500, lower_lim=0, upper_lim=1)
        # Exponential
        est01.append(estimator.estimate_mean(dist_creator.exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential"))
        # Weibull
        est02.append, est03.append = estimator.estimate_m_n(dist_creator.weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull")
        # Normal
        normal_arr = dist_creator.normal_dist(arr, mean=42000, std=663)
        est04.append(estimator.estimate_mean(normal_arr, mean=42000, dist_name="Normal"))
        est05.append(estimator.estimate_std(normal_arr, std=663, dist_name="Normal"))
        # Log Normal
        log_normal_arr = dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2)
        est06.append(estimator.estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal"))
        est07.append(estimator.estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal"))
        # Gumbel
        est09.append(estimator.estimate_mean(dist_creator.gumbel_dist(arr, mean=65000, std=370), mean=65000, dist_name="Gumbel"))
        est10.append(estimator.estimate_std(dist_creator.gumbel_dist(arr, mean=65000, std=370), std=370, dist_name="Gumbel"))
    # do for all and calc - רווח בר סמך?
    heapq.nlargest(10, est01)
    heapq.nsmallest(10, est01)

# Estimate with 50 numbers - halton
estimate_distribution_parameters_halton(50)
# Estimate with 200 numbers - halton
estimate_distribution_parameters_halton(200)
# Estimate with 500 numbers - halton
estimate_distribution_parameters_halton(500)
