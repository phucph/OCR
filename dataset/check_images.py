import util

ic15_root_dir = './data/icdar2015/'
ic15_train_data_dir = ic15_root_dir + 'train_images/'
ic15_train_gt_dir = ic15_root_dir + 'train_gts/'
ic15_test_data_dir = ic15_root_dir + 'ch4_test_images/'
ic15_test_gt_dir = ic15_root_dir + 'ch4_test_images_gts/'
data_dirs = [ic15_train_data_dir]
gt_dirs = [ic15_train_gt_dir]

for data_dir, gt_dir in zip(data_dirs, gt_dirs):
    img_names = util.io.ls(data_dir, '.jpg')
    img_names.extend(util.io.ls(data_dir, '.png'))
    # img_names.extend(util.io.ls(data_dir, '.gif'))

    img_paths = []
    gt_paths = []
    for idx, img_name in enumerate(img_names):
        img_path = data_dir + img_name
        img_paths.append(img_path)

        gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
        gt_path = gt_dir + gt_name
        gt_paths.append(gt_path)

