import os
import argparse
import numpy as np
import cv2
import copy
import dota_utils as util


def parse_args():
    parser = argparse.ArgumentParser(description='BBARegressive implementation')
    parser.add_argument('--src_path', type=str, default='imgs/', help='Src data directory')
    parser.add_argument('--dst_path', type=str, default='splitted_imgs/', help='Dst data directory')
    parser.add_argument('--phase', type=str, default='train', help='Phase directory')
    args = parser.parse_args()
    return args


class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=100,
                 subsize=1024,
                 ext='.png'):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
    def saveimagepatches(self, img, subimgname, left, up, ext='.png'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        cv2.imwrite(outdir, subimg)

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()
        all_labels = []
        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]
        
        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up)
                all_labels.append(subimgname)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
        return all_labels

    def splitdata(self, rate):
        
        imagelist = util.GetFileFromThisRootDir(self.srcpath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        imagenames = [name for name in imagenames if name.startswith('P')]
        all_labels = []
        for name in imagenames:
            labels = self.SplitSingle(name, rate, self.ext)
            all_labels.extend(labels)
        return all_labels

    def savesplitdata(self, data, fname):
        with open(os.path.join(self.outpath, fname), 'w') as file:
            for row in data:
                file.write(row+'\n')

if __name__ == '__main__':
    args = parse_args()
    split = splitbase(args.src_path, args.dst_path)
    test_list = split.splitdata(1)
    split.savesplitdata(test_list, 'test.txt')
