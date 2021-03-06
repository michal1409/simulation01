import math

import numpy


class DistributionCreator:

    def weibull_dist(self, num_array, n, m):
        ret = [n * numpy.power((-1 * math.log(num)) , (1 / m)) for num in num_array]
        return ret

    def exponential_dist(self, num_array, mean):
        ret = [math.log(num) * -1 * mean for num in num_array]
        return ret

    def normal_dist(self, num_array, mean, std):
        ret = []
        pi = math.pi
        i = 0
        while i < len(num_array) - 1:
            s1 = num_array[i]
            s2 = num_array[i + 1]
            ret.append((math.sqrt(-2 * math.log(s1)) * math.sin(2 * pi * s2)) * std + mean)
            ret.append((math.sqrt(-2 * math.log(s1)) * math.cos(2 * pi * s2)) * std + mean)
            i = i + 2
        return ret

    def logarithmic_normal_dist(self, num_array, mean, std):
        pi = numpy.pi
        ret = []
        i = 0
        num_array = numpy.float32(num_array)
        while i < len(num_array) - 1:
            s1 = num_array[i]
            s2 = num_array[i + 1]
            t1 = numpy.power(numpy.e , (numpy.sqrt(-1 * numpy.log(s1)) * numpy.sin(2 * pi * s2))) * std + mean
            t2 = numpy.power(numpy.e ,(numpy.sqrt(-1 * numpy.log(s1)) * numpy.cos(2 * pi * s2))) * std + mean
            ret.append(t1)
            ret.append(t2)
            i = i + 2
        return ret

    def gumbel_dist(self, num_array, mean, std):
        b = 0.78 * std
        ret = [mean - b * math.log(-1 * math.log(num)) for num in num_array]
        return ret

