class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/evg/Documents/All_segmentations/Qubvel'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'own_dataset':
            return '/home/evg/Documents/All_segmentations/detectron2_instance_segmentation_demo/data_slav_kuntsevo'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
