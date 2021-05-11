class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/tl32rodan/Desktop/109_2/AICup2021/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return '/home/tl32rodan/Desktop/109_2/AICup2021/benchmark/benchmark_RELEASE/' # folder that contains dataset/.
        elif database == 'cityscapes':
            return '/path/to/Segmentation/cityscapes/'         # foler that contains leftImg8bit/
        elif database == 'coco':
            return '/path/to/datasets/coco'        
        elif database == 'aicup':
            return '/home/tl32rodan/Desktop/109_2/AICup2021/data'        
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
