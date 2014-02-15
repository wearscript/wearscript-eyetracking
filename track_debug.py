from track import PARAMS, pupil_iter, debug_iter
import argparse
import time
import json
import cv2
import numpy as np

def warp_point(x, y):
    h = np.array([[  4.51756113e+00,  -1.24393424e+00,  -1.47805029e+02],
                  [ -1.76991625e+00,   5.85072303e+00,  -5.00441219e+02],
                  [ -7.75902188e-04,  -3.62933021e-04,   1.00000000e+00]])
    h = np.array([[  6.85043553e+00,  -2.23173495e+00,  -3.55373085e+02],
                  [  2.54252713e-01,   9.73267729e+00,  -1.99367793e+03],
                  [  1.58017218e-04,  -6.66932012e-04,   1.00000000e+00]])
    p = np.array([x, y, 1.])
    p = np.dot(p, h.T)
    return p[:2] / p[2]

def warp_point_multi(x, y):
    data = [[[[4.353886672622062, 0.868332552444964, -0.0020775858326677046], [-1.1738040364582627, 15.442211291452802, 0.0025395538654046416], [-72.44959374449643, -3766.6176161959647, 1.0]], [259.51527976989746, 267.3915087381999]], [[[-3.926077053689926, -3.1769177035464544, -0.0018182804365206794], [-1.4789208054173177, -14.508016301419929, -0.0037964476486834404], [1059.7883431755865, 4611.021694528221, 1.0]], [268.55883916219074, 327.21007537841797]], [[[2.845873808595234, -0.08095320196065668, -0.0004361465050138748], [-1.1435011488670397, 8.898048017051067, -0.0003226622910547123], [337.63818131472846, -1839.9936577450208, 1.0]], [319.3537775675456, 257.9471130371094]], [[[3.572863425005356, -1.0438020756862887, -0.001325685180259236], [0.7773949654677705, 13.337425390084338, 0.0016352473166703016], [-93.34012789963835, -2454.619847623105, 0.9999999999999999]], [340.18710645039874, 319.18003336588544]], [[[3.2530812182446605, -0.8920043540640313, -0.00048152696554550604], [-0.6215207565132795, 10.31778420855704, 0.0001341037816716763], [518.6413649807852, -1856.3304165365184, 1.0]], [383.8747749328613, 259.5265115102132]], [[[-0.23815448443049778, -1.126701991017355, -0.0009353348664464785], [-2.1166735241551224, 2.149436723591208, -0.001033806125547819], [1334.708604897373, 169.12703649471612, 1.0]], [396.45519256591797, 316.63880666097003]]]
    centers = []
    hs = []
    for h, center in data:
        hs.append(np.array(h))
        centers.append(np.array(center))
    centers = np.asfarray(centers)
    print(centers)
    p = np.array([x, y])
    print(p)
    d = centers - p
    print(d)
    print(np.sum(d * d, 1))
    ind = np.argmin(np.sum(d * d, 1))
    print(ind)
    p = np.array([x, y, 1.])
    p = np.dot(p, hs[ind])
    return p[:2] / p[2]


def debug(**kw):
    run = [None]
    print(PARAMS)
    kw.update(PARAMS)
    track_frame = cv2.imread('20140215_113524_818.jpg')
    print(track_frame.shape)
    track_frame = cv2.resize(track_frame, (track_frame.shape[1] / 2, track_frame.shape[0] / 2))
    print(track_frame.shape)
    while run[0] != 'QUIT':
        run = [None]
        for data in pupil_iter(debug=True, **kw):
            box = data[0]
            if box and 1:
                p = warp_point(box[0][0], box[0][1]) / 2
                print(p)
                track_frame_copy = track_frame.copy()
                cv2.circle(track_frame_copy, (int(np.round(p[0])), int(np.round(p[1]))), 10, (0, 255, 0))
                cv2.imshow("Track", track_frame_copy)
            run[0] = debug_iter(*data)
            if run[0]:
                break

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump')
    parser.add_argument('--load')
    parser.add_argument('--calib')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    vargs = vars(args)    
    debug(**vargs)

