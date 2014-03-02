import glob
import argparse
import json
import matplotlib
matplotlib.use('wx')
import sys
import cv2
import random
import scipy.optimize
import numpy as np
from match import ImageMatch, ImagePoints, image_size, click_points, fit_homography

def level_finder(stream, inset, max_time=.7, offset_time=None):
    levels = []
    cur_level = []
    first_time = None
    last_time = None
    samples = 0
    for sample_time, sample in stream:
        samples += 1
        if first_time is None:
            first_time = sample_time
        last_time = sample_time
        sample_time = sample_time - offset_time
        if cur_level and (not inset(cur_level[0][1], sample) or sample_time - cur_level[-1][0] > max_time):
            levels.append(cur_level)
            cur_level = []
        cur_level.append((sample_time, sample))
    dur = last_time - first_time
    print('Duration[%f] Samples[%f] FPS[%f]' % (dur, samples, samples / dur))
    return levels

def levels_times(levels, min_level):
    return [(x[-1][0] - x[0][0], x[0][0], x[-1][0]) for x in levels if x[-1][0] - x[0][0] > min_level]

def levels_dist(lt0, lt1, offset=0):
    overlap = 0
    for _, start_time0, end_time0 in lt0:
        start_time0 += offset
        end_time0 += offset
        for _, start_time1, end_time1 in lt1:
            if start_time0 <= start_time1 <= end_time0:
                overlap += min(end_time0, end_time1) - start_time1
            elif start_time1 <= start_time0 <= end_time1:
                overlap += min(end_time0, end_time1) - start_time0
    duration = min(lt0[-1][1] - lt0[0][1], lt1[-1][1] - lt1[0][1])
    return overlap / duration
                    

def select_point(world_image):
    return click_points(1, cv2.imread(world_image))[0]

def warp_homographies(time_image_paths, points, match):
    world_image = random.choice(time_image_paths)[1]
    x, y = select_point(world_image)
    points_world = points(open(world_image).read())
    for sample_time, image_path in time_image_paths:
        points0 = points(open(image_path).read())
        h = match(points_world, points0)
        sample_image = cv2.imread(image_path)
        z = np.dot(h, np.array([x, y, 1]))
        z = z[:2] / z[2]
        print(h)
        cv2.circle(sample_image, (int(z[0]), int(z[1])), 5, (255, 0, 0))
        cv2.imshow('0', sample_image)
        cv2.waitKey(15)
        yield sample_time, z[0], z[1]

def align_time_points(time_points_world, time_points_eye, time_delta=.1):
    pairs = []
    for t0, x0, y0 in time_points_world:
        dt, x1, y1 = min((np.abs(t0 - t1), x1, y1) for t1, x1, y1 in time_points_eye)
        print(dt)
        if dt < time_delta:
            pairs.append([x0, y0, x1, y1])
    return pairs
    

def calibrate(dump, calibdump, pupil_dist_thresh=5, h_dist_thresh=30, min_level=.5):
    match = ImageMatch()
    points = ImagePoints()
    pupil_dist_thresh = pupil_dist_thresh ** 2

    def iter1_paths():
        for x in sorted(glob.glob(calibdump + '/*.jpg')):
            yield float(x.rsplit('/', 1)[1].rsplit('.', 1)[0]), x

    def iter0():
        for x in sorted(glob.glob(dump + '/*.js')):
            yield float(x.rsplit('/', 1)[1].rsplit('.', 1)[0]), json.load(open(x))
    def inset0(x, y):
        dx = x['x'] - y['x']
        dy = x['y'] - y['y']
        return dx * dx + dy + dy < pupil_dist_thresh

    def iter1():
        for x in sorted(glob.glob(calibdump + '/*.jpg')):
            image_data = open(x).read()
            sz = image_size(image_data)
            yield float(x.rsplit('/', 1)[1].rsplit('.', 1)[0]), (points(image_data), np.array([sz[0] / 2, sz[1] / 2, 1.]))
    def inset1(x, y):
        h = match(x[0], y[0])
        z = np.dot(h, x[1])
        z = z[:2] / z[2]
        #print((z, x[1][:2]))
        d = z - x[1][:2]
        d = np.sum(d * d)
        #print('Dist: ' + str(d))
        #print(h)
        return d < h_dist_thresh
    # Try to align the "levels" such that they have maximum intersection to identify time offset
    offset_time = min(iter0().next()[0], iter1().next()[0])
    lt1 = levels_times(level_finder(iter1(), inset1, offset_time=offset_time), min_level=min_level)
    lt0 = levels_times(level_finder(iter0(), inset0, offset_time=offset_time), min_level=min_level)
    print(lt0)
    print(lt1)
    print(levels_dist(lt0, lt1))
    print(levels_dist(lt0, lt0))
    print(levels_dist(lt1, lt1))
    for x in np.arange(-2, 2, .1):
        print((levels_dist(lt0, lt1, x), x))
    offset = scipy.optimize.fmin(lambda x: -levels_dist(lt0, lt1, x), 0)[0]
    time_points_world = list(warp_homographies(list(iter1_paths()), points, match))
    time_points_eye = [(t + offset, data['x'], data['y']) for t, data in iter0()]
    pairs = np.array(align_time_points(time_points_world, time_points_eye))
    print(fit_homography(pairs).ravel().tolist())
    print('offset[%f]' % offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('calibdump')
    args = parser.parse_args()
    calibrate(args.dump, args.calibdump)

if __name__ == '__main__':
    main()
