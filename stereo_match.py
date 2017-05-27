# -*- coding=utf8 -*-
from __future__ import print_function
import numpy as np
import cv2


ply_header='''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1,3)
    colors = colors.reshape(-1,3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt = '%f %f %f %d %d %d ')


def main():
    frame_left = cv2.imread('1.jpg')
    frame_right = cv2.imread('1.jpg')
    frame_left=frame_left[:,:1384,:]
    frame_right = frame_right[:, 1384:, :]
    print (frame_left.shape)
    print (frame_right.shape)

    #конвертирую изображения в градации серого
    #вычисляю "разницу" между левым и правым изображением
    l = cv2.pyrDown(frame_left)
    r = cv2.pyrDown(frame_right)
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp

    stereo = cv2.StereoSGBM(
        minDisparity=min_disp,
        numDisparities=num_disp,
        SADWindowSize=16,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = stereo.compute(l,r).astype(np.float32) / 16.0
    h,w = l.shape[:2]
    f = 0.8*w
    Q = np.float32([[1, 0, 0,-0.5*w],
                    [0,-1, 0, 0.5*h],
                    [0, 0, 0,   -f],
                    [0, 0, 1,    0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(l,cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    #fuck
    cv2.imshow('l', l)
    cv2.imshow('d', (disp-min_disp)/num_disp)
    if cv2.waitKey() & 0xFF == 27:
        cv2.destroyAllWindows()
    elif cv2.waitKey() & 0xFF == ord('s'):
        out_points = points[mask]
        out_color = colors[mask]
        out_fn = 'out_.ply'
        write_ply(out_fn, out_points, out_color)
        print ('%s saved' % out_fn)







if __name__ == '__main__':
    main()