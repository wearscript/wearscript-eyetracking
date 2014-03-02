import numpy as np
import glob
import argparse
import json
import cv2

OFFSET = -0.533500

h = np.array([-2.3313015604932312, -2.9887827128043156, 714.575406341617, -0.6480698173994087, -4.043698204243812, 677.0861184808796, -0.0015529308094834155, -0.010996420857239788, 1.0]).reshape((3, 3))
#h = np.mat(h).I.A

def align_world_eye_points(world_paths, time_points_eye, time_delta=2):
    pairs = []
    for t0, image_path in world_paths:
        dt, x1, y1 = min((np.abs(t0 - t1), x1, y1) for t1, x1, y1 in time_points_eye)
        print(dt)
        if dt < time_delta:
            z = np.dot(h, [x1, y1, 1])
            z = z[:2] / z[2]
            sample_image = cv2.imread(image_path)
            cv2.circle(sample_image, (int(z[0]), int(z[1])), 5, (255, 0, 0))
            cv2.imshow('0', sample_image)
            cv2.waitKey(500)

def combine(dump, calibdump, pupil_dist_thresh=5, h_dist_thresh=30, min_level=.5):
    def iter1_paths():
        for x in sorted(glob.glob(calibdump + '/*.jpg')):
            yield float(x.rsplit('/', 1)[1].rsplit('.', 1)[0]), x

    def iter0():
        for x in sorted(glob.glob(dump + '/*.js')):
            yield float(x.rsplit('/', 1)[1].rsplit('.', 1)[0]), json.load(open(x))
    world_paths = list(iter1_paths())
    time_points_eye = [(t + OFFSET, data['x'], data['y']) for t, data in iter0()]
    align_world_eye_points(world_paths, time_points_eye)
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('calibdump')
    args = parser.parse_args()
    combine(args.dump, args.calibdump)

if __name__ == '__main__':
    main()
