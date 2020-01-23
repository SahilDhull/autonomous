function [ det_boxes, det_probs, det_classes ] = object_detection( classifier_obj, image )
%OBJECT_DETECTION Summary of this function goes here
%   Detailed explanation goes here

[image_h, image_w, ~] = size(image);
temp_img_flip = flip(image, 3);  % Convert from RGB to BGR
temp_img_permuted = permute(temp_img_flip, [3,2,1]);
image_1d = reshape(temp_img_permuted, 1, []);
detections = classifier_obj.do_object_detection_on_raw_data_for_matlab(py.list(image_1d), uint32(image_w), uint32(image_h));

det_boxes = [];
det_classes = double(py.array.array('d', detections{3}));
for ii = 1:length(det_classes)
    det_boxes = [det_boxes; double(py.array.array('d', detections{1}{ii}))];
end
det_probs = double(py.array.array('d', detections{2}));

end

