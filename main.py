from numpy import random
import numpy
import chaospy as cy
import heapq


######### destribution functions ###########
def gen_from_seed(seed,array_size,lower_lim,upper_lim):
    random.seed(seed)
    ret = random.uniform(lower_lim,upper_lim,array_size)
    return ret


def exponential_dist(num_array, mean):
    ret = [numpy.log(num) * -1 * mean for num in num_array]
    return ret


def weibull_dist(num_array, n, m):
    ret = [n * numpy.power((-1 * numpy.log(num)),(1/m)) for num in num_array]
    return ret


def normal_dist(num_array, mean, std):
    pi = numpy.pi
    ret = []
    i = 0
    while i < len(num_array) - 1:
        s1 = num_array[i]
        s2 = num_array[i+1]
        ret.append((numpy.sqrt(-2 * numpy.log(s1)) * numpy.sin(2 * pi * s2))*std + mean)
        ret.append((numpy.sqrt(-2 * numpy.log(s1)) * numpy.cos(2 * pi * s2))*std + mean)
        i = i + 2
    return ret


def logarithmic_normal_dist(num_array, mean, std):
    pi = numpy.pi
    ret = []
    i = 0
    while i < len(num_array) - 1:
        s1 = num_array[i]
        s2 = num_array[i + 1]
        ret.append((numpy.power(numpy.e, numpy.sqrt(-1 * numpy.log(s1)) * numpy.sin(2*pi * s2)))*std + mean)
        ret.append((numpy.power(numpy.e, numpy.sqrt(-1 * numpy.log(s1)) * numpy.cos(2*pi * s2)))*std + mean)
        i = i + 2
    return ret


def gumbel_dist(num_array, m, std):
    b = 0.78 * std
    ret = [m - b * numpy.log( -1 * numpy.log(num)) for num in num_array]
    return ret


######### estimation functions ###########
def estimate_mean(dist_arr, mean, dist_name,onlyvalue):
    est_m = numpy.average(dist_arr)
    if onlyvalue:
        return est_m
    else:
        return f"Estimate for {dist_name}: The mean estimation is: {est_m} the real value is: {mean}"


def estimate_std(dist_arr,std,dist_name,onlyvalue):
    est_s = numpy.std(dist_arr)
    if onlyvalue:
        return est_s
    else:
        return f"Estimate for {dist_name}: The std estimation is: {est_s} the real value is: {std}"


def estimate_m_n(dist_arr, m, n, dist_name,onlyvalue):
    # n:
    sum_x_in_power_m = numpy.sum([numpy.power(x, m) for x in dist_arr])
    est_n = numpy.power(1/len(dist_arr) * sum_x_in_power_m, (1/m))
    # m:
    sum_x_times_ln = numpy.sum([numpy.power(x, m) * numpy.log(x) for x in dist_arr])
    denominator = sum_x_times_ln/sum_x_in_power_m - (1/len(dist_arr)) * numpy.sum([numpy.log(x) for x in dist_arr])
    est_m = 1/denominator
    if onlyvalue:
        return est_m,est_n
    else:
        return f"Estimate for {dist_name}: The m estimation is: {est_m} the real value is: {m} \n The n estimation is: {est_n} the real value is: {n}"

####################### prints for delivery #####################
def estimate_distribution_parameters(seed, array_size):
    print()
    print(f"Estimate with {array_size}:")
    arr = gen_from_seed(seed=seed, array_size=array_size, lower_lim=0, upper_lim=1)
    # Exponential
    print(estimate_mean(exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential",onlyvalue=True))
    # Weibull
    print(estimate_m_n(weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull",onlyvalue=True))
    # Normal
    normal_arr = normal_dist(arr, mean=42000, std=663)
    print(estimate_mean(normal_arr, mean=42000, dist_name="Normal",onlyvalue=True))
    print(estimate_std(normal_arr, std=663, dist_name="Normal",onlyvalue=True))
    # Log Normal
    log_normal_arr = logarithmic_normal_dist(arr, mean=11, std=1.2)
    print(estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal",onlyvalue=True))
    print(estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal",onlyvalue=True))
    # Gumbel
    print(estimate_mean(gumbel_dist(arr, m=65000, std=370), mean=65000, dist_name="Gumbel",onlyvalue=True))
    print(estimate_std(gumbel_dist(arr, m=65000, std=370), std=370, dist_name="Gumbel",onlyvalue=True))

def gen_from_halton(array_size,lower_lim,upper_lim):
    uniform = cy.Uniform(lower_lim,upper_lim)
    samples = uniform.sample(array_size, rule="halton")
    return samples

def estimate_distribution_parameters_halton(array_size):
    print()
    print(f"Estimate with {array_size}:")
    arr = gen_from_halton(array_size=array_size, lower_lim=0, upper_lim=1)
    # Exponential
    print(estimate_mean(exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential",onlyvalue=True))
    # Weibull
    print(estimate_m_n(weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull",onlyvalue=True))
    # Normal
    normal_arr = normal_dist(arr, mean=42000, std=663)
    print(estimate_mean(normal_arr, mean=42000, dist_name="Normal",onlyvalue=True))
    print(estimate_std(normal_arr, std=663, dist_name="Normal",onlyvalue=True))
    # Log Normal
    log_normal_arr = logarithmic_normal_dist(arr, mean=11, std=1.2)
    print(estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal",onlyvalue=True))
    print(estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal",onlyvalue=True))
    # Gumbel
    print(estimate_mean(gumbel_dist(arr, m=65000, std=370), mean=65000, dist_name="Gumbel",onlyvalue=True))
    print(estimate_std(gumbel_dist(arr, m=65000, std=370), std=370, dist_name="Gumbel",onlyvalue=True))



###############tests#################
def test01():
    print("hello")
    print(gen_from_seed(10,10,0,1))
    print(gen_from_seed(10,10,0,1))

def test02(): # test function part A
    arr = gen_from_seed(seed=500, array_size=10, lower_lim=0, upper_lim=1)
    a = exponential_dist(arr, mean=120000)
    b = weibull_dist(arr, n=7600, m=1.2)
    c = normal_dist(arr, mean=42000, std=663)
    d = logarithmic_normal_dist(arr, mean=11, std=1.2)
    e = gumbel_dist(arr, m=65000, std=370)

############### answers #################

#A
# generate numbers from seed
gen_from_seed(1)

#B
# Estimate with 500 numbers - good estimation
estimate_distribution_parameters(1, 500)

#C
# Estimate with 500 numbers and same seed - will be the same
estimate_distribution_parameters(1, 500)
# Estimate with different seed - the estimations is still good, the outcome is not exactly the same numbers
estimate_distribution_parameters(5, 500)
# Estimate with 1000 numbers and second seed - is it better? is it slow conversions?(1000 better then 500)
estimate_distribution_parameters(10, 1000)

#D
est01= [], est02= [], est03= [],est04= [],est05= [],est06= [],est07= [],est08= [],est09= [],est10= []
for i in range(100):
    new_seed = random.randint()
    arr = gen_from_seed(seed=new_seed, array_size=500, lower_lim=0, upper_lim=1)
    # Exponential
    est01.append(estimate_mean(exponential_dist(arr, mean=120000), mean=120000, dist_name="Exponential"))
    # Weibull
    est02.append,est03.append = estimate_m_n(weibull_dist(arr, n=76000, m=1.2), m=1.2, n=76000, dist_name="Weibull")
    # Normal
    normal_arr = normal_dist(arr, mean=42000, std=663)
    est04.append(estimate_mean(normal_arr, mean=42000, dist_name="Normal"))
    est05.append(estimate_std(normal_arr, std=663, dist_name="Normal"))
    # Log Normal
    log_normal_arr = logarithmic_normal_dist(arr, mean=11, std=1.2)
    est06.append(estimate_mean(log_normal_arr, mean=11, dist_name="Logarithmic Normal"))
    est07.append(estimate_std(log_normal_arr, std=1.2, dist_name="Logarithmic Normal"))
    # Gumbel
    est09.append(estimate_mean(gumbel_dist(arr, m=65000, std=370), mean=65000, dist_name="Gumbel"))
    est10.append(estimate_std(gumbel_dist(arr, m=65000, std=370), std=370, dist_name="Gumbel"))
#do for all and calc - רווח בר סמך?
heapq.nlargest(10, est01)
heapq.nsmallest(10,est01)

# Estimate with 50 numbers - halton
estimate_distribution_parameters_halton(50)
# Estimate with 200 numbers - halton
estimate_distribution_parameters_halton(200)
# Estimate with 500 numbers - halton
estimate_distribution_parameters_halton(500)

