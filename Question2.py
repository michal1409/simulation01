import math

import ks as ks
import numpy

from DistributionCreator import DistributionCreator
from Estimator import Estimator
from Question1 import Main
from scipy import stats

print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Question2")
print("--------------------------------------------------------------------------------------------------------------------------------------------")
# create 500 probabilities
arr = Main.gen_from_seed(seed=1, array_size=500, lower_lim=0, upper_lim=1)

# create a vector with 500 numbers from each system matching to each one of the probabilities
dist_creator = DistributionCreator()
blade = dist_creator.normal_dist(arr, mean=42000, std=663)
gearbox = dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2)
generator = dist_creator.weibull_dist(arr, n=76000, m=1.2)
yaw = dist_creator.gumbel_dist(arr, mean=65000, std=370)
pitch_control = dist_creator.normal_dist(arr, mean=84534, std=506)
brake = dist_creator.exponential_dist(arr, mean=120000)
lubrication = dist_creator.weibull_dist(arr, n=66000, m=1.3)
electrical = dist_creator.weibull_dist(arr, n=35000, m=1.5)
frequency = dist_creator.exponential_dist(arr, mean=45000)

matrix = [blade, gearbox, generator, yaw, pitch_control, brake, lubrication, electrical, frequency]
min_values = numpy.amin(matrix, axis=0)
#print(f"min vals: {min_values}")
#print(f"gear: {gearbox}")
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

################### B ###########################3

bins_arr1,bin_edge1 = numpy.histogram(min_values,bins= [5,11.2,11.4,11.5,11.55,11.65,11.75,11.8,11.9,12,12.1,12.2,12.3,12.35,12.4,12.45039336,12.55,12.6,12.7,12.8,12.9,13.4,13.73844492,15.02649648,16.31454804]
)
print(f"bins array : {bins_arr1} , bins edges: {bin_edge1}")
print(f"blade : {len(blade)} , nimval : {len(min_values)}")

bins_arr1,bin_edge1 = numpy.histogram(min_values,int(numpy.sqrt(500)))

blade_ks = stats.ks_2samp(min_values,blade).statistic

blade_anderson = stats.anderson_ksamp([min_values,blade]).statistic

bins_blade,bin_edge2 = numpy.histogram(blade,bin_edge1)
blade_chi = stats.chisquare(bins_blade, bins_arr1,2).statistic

print(f"blade ks:{blade_ks} , anderson:{blade_anderson} , chi:{blade_chi}")

gearbox_ks = stats.ks_2samp(min_values,gearbox).statistic
gearbox_anderson = stats.anderson_ksamp([min_values,gearbox]).statistic
bins_gearbox,bin_edge2 = numpy.histogram(gearbox,bin_edge1)
gearbox_chi = stats.chisquare(bins_gearbox,bins_arr1,2).statistic
print(f"gearbox ks:{gearbox_ks} , anderson:{gearbox_anderson} , chi:{gearbox_chi}")

generator_ks = stats.ks_2samp(min_values,generator).statistic
generator_anderson = stats.anderson_ksamp([min_values,generator]).statistic
bins_generator,bin_edge2 = numpy.histogram(generator,bin_edge1)
generator_chi = stats.chisquare(bins_generator,bins_arr1,2).statistic
print(f"generator ks:{generator_ks} , anderson:{generator_anderson} , chi:{generator_chi}")

yaw_ks = stats.ks_2samp(min_values,yaw).statistic
yaw_anderson = stats.anderson_ksamp([min_values,yaw]).statistic
bins_yaw,bin_edge2 = numpy.histogram(yaw,bin_edge1)
yaw_chi = stats.chisquare(bins_yaw,bins_arr1,2).statistic
print(f"yaw ks:{yaw_ks} , anderson:{yaw_anderson} , chi:{yaw_chi}")

pitch_control_ks = stats.ks_2samp(min_values,pitch_control).statistic
pitch_control_anderson = stats.anderson_ksamp([min_values,pitch_control]).statistic
bins_pitch_control,bin_edge2 = numpy.histogram(pitch_control,bin_edge1)
pitch_control_chi = stats.chisquare(bins_pitch_control,bins_arr1,2).statistic
print(f"pitch_control ks:{pitch_control_ks} , anderson:{pitch_control_anderson} , chi:{pitch_control_chi}")

brake_ks = stats.ks_2samp(min_values,brake).statistic
brake_anderson = stats.anderson_ksamp([min_values,brake]).statistic
bins_brake,bin_edge2 = numpy.histogram(brake,bin_edge1)
brake_chi = stats.chisquare(bins_brake,bins_arr1,1).statistic
print(f"brake ks:{brake_ks} , anderson:{brake_anderson} , chi:{brake_chi}")

lubrication_ks = stats.ks_2samp(min_values,lubrication).statistic
lubrication_anderson = stats.anderson_ksamp([min_values,lubrication]).statistic
bins_lubrication,bin_edge2 = numpy.histogram(lubrication,bin_edge1)
lubrication_chi = stats.chisquare(bins_lubrication,bins_arr1,2).statistic
print(f"lubrication ks:{lubrication_ks} , anderson:{lubrication_anderson} , chi:{lubrication_chi}")

electrical_ks = stats.ks_2samp(min_values,electrical).statistic
electrical_anderson = stats.anderson_ksamp([min_values,electrical]).statistic
bins_electrical,bin_edge2 = numpy.histogram(electrical,bin_edge1)
electrical_chi = stats.chisquare(bins_electrical,bins_arr1,2).statistic
print(f"electrical ks:{electrical_ks} , anderson:{electrical_anderson} , chi:{electrical_chi}")

frequency_ks = stats.ks_2samp(min_values,frequency).statistic
frequency_anderson = stats.anderson_ksamp([min_values,frequency])
bins_frequency,bin_edge2 = numpy.histogram(electrical,bin_edge1)
frequency_chi = stats.chisquare(bins_frequency,bins_arr1,1).statistic
print(f"frequency ks:{frequency_ks} , anderson:{frequency_anderson.statistic} , chi:{frequency_chi}")

print(f"anderson_critical =  {frequency_anderson.critical_values[-3]}")
ks_critical = 1.36 / math.sqrt(500)
print(f"{ks_critical}")
'''
blade = numpy.array(dist_creator.normal_dist(arr, mean=42000, std=663))
gearbox = numpy.array(dist_creator.logarithmic_normal_dist(arr, mean=11, std=1.2))
generator = numpy.array(dist_creator.weibull_dist(arr, n=76000, m=1.2))
yaw = numpy.array(dist_creator.gumbel_dist(arr, mean=65000, std=370))
pitch_control = numpy.array(dist_creator.normal_dist(arr, mean=84534, std=506))
brake = numpy.array(dist_creator.exponential_dist(arr, mean=120000))
lubrication = numpy.array(dist_creator.weibull_dist(arr, n=66000, m=1.3))
electrical = numpy.array(dist_creator.weibull_dist(arr, n=35000, m=1.5))
frequency = numpy.array(dist_creator.exponential_dist(arr, mean=45000))
'''