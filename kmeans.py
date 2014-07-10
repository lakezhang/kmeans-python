#encoding utf-8
import math
import sys
import random
import copy
from collections import defaultdict

def distance(a, b, gauss=True):
    """
        Euclidean distance or gauss kernel
    """
    dim = len(a)
    
    _sum = 0.0
    for dimension in xrange(dim):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq

    if gauss :
        dis = 1 - math.exp(_sum * -1/2)
    else :
        dis = math.sqrt(_sum)

    return dis

def load_file(file_name) :
    """
        Load dataset, for iris dataset
        @return data, count
        count : the records num 
        data : {'f' : feature list , 'c' : cluster id, 'l' : origin cluster label}
    """
    data = {}

    count = 0
    with open(file_name) as fp :
        for line in fp :
            line = line.strip()
            if len(line) < 1 :
                continue

            tokens = line.split(',')
            if len(tokens) < 2 :
                continue
            #this last column is label
            data[count] = {'f' : [float(x) for x in tokens[:-1]], 'c' : -1, 'l' : tokens[-1]}
            count += 1

    return data, count

def init_centers(data, count, center_num) :
    """
        Random init centers
    """
    centers = set()

    if center_num > count / 2 :
        raise Exception("too many centers!")

    while len(centers) < center_num :
        c = random.randint(0, count - 1)
        centers.add(c)

    center_feature = {}
    for i in centers :
        center_feature[i] = data[i]['f']

    return center_feature

def update_centers(data) :
    """
        Update centers
    """
    center_feature = {}
    center_nums = defaultdict(lambda : 0)
    for k, v in data.iteritems() :
        center_nums[v['c']] += 1

        if not v['c'] in center_feature :
            center_feature[v['c']] = copy.deepcopy(v['f'])
        else :
            center_feature[v['c']] = [center_feature[v['c']][i] + j for i, j in enumerate(v['f'])]

    for k, v in center_feature.iteritems() :
        center_feature[k] = [float(i) / center_nums[k] for i in v]

    return center_feature

def kmeans(file_name, center_num, gauss, iterate_num, decrease_threshold) :
    """
        kmeans
    """
    data, count = load_file(file_name)
    center_feature = init_centers(data, count, center_num)
    total_distance, last_distance = 0.0, 0.0
    count = 1

    while True :  
        last_distance = total_distance
        total_distance = 0.0
        for k,  v in data.iteritems() :
            min_dis = sys.float_info.max
            step_cluster = -1

            for i, j in center_feature.iteritems() :
                dis = distance(v['f'], j, gauss)
                if dis < min_dis :
                    min_dis = dis
                    step_cluster = i

            total_distance += min_dis
            data[k]['c'] = step_cluster

        center_feature = update_centers(data)

        decrease = last_distance - total_distance if count != 1 else total_distance

        print "STEP: %d, TOTAL_DISTANCE: %.4f, DESCREASE: %.4f" % (count, total_distance, decrease)
        
        if decrease < decrease_threshold :
            break

        count += 1
        if count > iterate_num :
            break

    evaluate(data)

def evaluate(data) :
    """
        evaluate precision
    """
    stat = defaultdict(lambda : defaultdict(lambda : 0))
    for k, v in data.iteritems() :
        stat[v['c']][v['l']] += 1

    wholecount = 0
    wholecorrect = 0
    for k, v in stat.iteritems() :
        allcount = 0
        maxj = 0
        maxi = ''
        for i, j in v.iteritems() :
            allcount += j
            if j > maxj :
                maxj = j
                maxi = i


        wholecorrect += maxj
        wholecount += allcount
        print "CLUSTER: %d ALLNUM: %d CORRECT: %d PRECISION: %.4f LABEL: %s" % (k, allcount, maxj, float(maxj) / allcount, maxi)

    print "ALLNUM: %d CORRECT: %d PRECISION: %.4f" % (wholecount, wholecorrect, float(wholecorrect) / wholecount)


if __name__ == '__main__':
    
    kmeans('data/iris.data', 3, False, 100, 0.01)
