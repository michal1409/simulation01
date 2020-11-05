import numpy

from DistributionCreator import DistributionCreator
from Estimator import Estimator
from Question1 import Main

print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Question2")
print("--------------------------------------------------------------------------------------------------------------------------------------------")
# create 500 probabilities
arr = Main.gen_from_seed(seed=1, array_size=10, lower_lim=0, upper_lim=1)

# create a vector with 500 numbers from each system matching to each one of the probabilities
dist_creator = DistributionCreator()
blade = numpy.array(dist_creator.normal_dist(arr, mean=42000, std=663))
gearbox = numpy.array(dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2))
generator = numpy.array(dist_creator.weibull_dist(arr, n=76000, m=1.2))
yaw = numpy.array(dist_creator.gumbel_dist(arr, mean=65000, std=370))
pitch_control = numpy.array(dist_creator.normal_dist(arr, mean=84534, std=506))
brake = numpy.array(dist_creator.exponential_dist(arr, mean=120000))
lubrication = numpy.array(dist_creator.weibull_dist(arr, n=66000, m=1.3))
electrical = numpy.array(dist_creator.weibull_dist(arr, n=35000, m=1.5))
frequency = numpy.array(dist_creator.exponential_dist(arr, mean=45000))

matrix = [blade, gearbox, generator, yaw, pitch_control, brake, lubrication, electrical, frequency]
min_values = numpy.amin(matrix, axis=0)

estimator = Estimator()

# turbine distribution estimation
turbine_mean_estimation = estimator.estimate_mean(dist_arr=min_values)
turbine_std_estimation = estimator.estimate_std(dist_arr=min_values)
print(f"turbine mean: {turbine_mean_estimation}")
print(f"turbine std: {turbine_std_estimation}")

blade_mean_estimation = estimator.estimate_mean(dist_arr=blade)
gearbox_mean_estimation = estimator.estimate_mean(dist_arr=gearbox)
yaw_mean_estimation = estimator.estimate_mean(dist_arr=yaw)
pitch_control_mean_estimation = estimator.estimate_mean(dist_arr=pitch_control)
brake_mean_estimation = estimator.estimate_mean(dist_arr=brake)
frequency_mean_estimation = estimator.estimate_mean(dist_arr=frequency)
print("other distributions mean")
print(blade_mean_estimation)
print(gearbox_mean_estimation)
print(yaw_mean_estimation)
print(pitch_control_mean_estimation)
print(brake_mean_estimation)
print(frequency_mean_estimation)

blade_std_estimation = estimator.estimate_std(dist_arr=blade)
gearbox_std_estimation = estimator.estimate_std(dist_arr=gearbox)
yaw_std_estimation = estimator.estimate_std(dist_arr=yaw)
pitch_control_std_estimation = estimator.estimate_std(dist_arr=pitch_control)

print("other distributions std")
print(blade_std_estimation)
print(gearbox_std_estimation)
print(yaw_std_estimation)
print(pitch_control_std_estimation)

print("Weibull parameters: m and n")
generator_m_n_estimation = estimator.estimate_m_n(dist_arr=generator, m=1.2)
turbine_m_n_estimation_gen_reference = estimator.estimate_m_n(dist_arr=min_values, m=1.2)
lubrication_m_n_estimation = estimator.estimate_m_n(dist_arr=lubrication, m=1.3)
turbine_m_n_estimation_lub_reference = estimator.estimate_m_n(dist_arr=min_values, m=1.3)
electrical_m_n_estimation = estimator.estimate_m_n(dist_arr=electrical, m=1.5)
turbine_m_n_estimation_elec_reference = estimator.estimate_m_n(dist_arr=min_values, m=1.5)

print(f"{generator_m_n_estimation} -> {turbine_m_n_estimation_gen_reference}")
print(f"{lubrication_m_n_estimation} -> {turbine_m_n_estimation_lub_reference}")
print(f"{electrical_m_n_estimation} -> {turbine_m_n_estimation_elec_reference}")
