temp_img = imread('webots3.png');
[ det_boxes, det_probs, det_classes ] = object_detection(classifier_obj, temp_img)


temp_img_flip = flip(temp_img, 3);  % Convert from RGB to BGR
temp_img_permuted = permute(temp_img, [1,2,3]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)

temp_img_permuted = permute(temp_img_flip, [1,3,2]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)

temp_img_permuted = permute(temp_img_flip, [2,1,3]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)

temp_img_permuted = permute(temp_img_flip, [2,3,1]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)

temp_img_permuted = permute(temp_img_flip, [3,1,2]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)

temp_img_permuted = permute(temp_img_flip, [3,2,1]);

%classifier_obj = start_object_detector();
dets = object_detection(classifier_obj, temp_img_permuted)


