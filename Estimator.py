import numpy

class Estimator:

    def estimate_m_n(self, dist_arr, m, n=None, dist_name=None, onlyvalue=True):
        # n:
        sum_x_in_power_m = numpy.sum([numpy.power(x, m) for x in dist_arr])
        est_n = numpy.power(1 / len(dist_arr) * sum_x_in_power_m, (1 / m))
        # m:
        sum_x_times_ln = numpy.sum([numpy.power(x, m) * numpy.log(x) for x in dist_arr])
        denominator = sum_x_times_ln / sum_x_in_power_m - (1 / len(dist_arr)) * numpy.sum([numpy.log(x) for x in dist_arr])
        est_m = 1 / denominator
        if onlyvalue:
            return est_m, est_n
        else:
            return f"Estimate for {dist_name}: The m estimation is: {est_m} the real value is: {m} \n The n estimation is: {est_n} the real value is: {n}"

    def estimate_mean(self, dist_arr, mean=None, dist_name=None, onlyvalue=True):
        est_m = numpy.average(dist_arr)
        if onlyvalue:
            return est_m
        else:
            return f"Estimate for {dist_name}: The mean estimation is: {est_m} the real value is: {mean}"

    def estimate_std(self, dist_arr, std=None, dist_name=None, onlyvalue=True):
        est_s = numpy.std(dist_arr)
        if onlyvalue:
            return est_s
        else:
            return f"Estimate for {dist_name}: The std estimation is: {est_s} the real value is: {std}"

