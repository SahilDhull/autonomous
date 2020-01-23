desktop;

% Webots called this script.
% Here we can call Simulink model or anything else.


TARGET_SPEED = 30;  % km/h
CONTROL_P_GAIN = 0.1;

wbu_driver_init(); % We must first initialize driver library

% The following are to set the car gear to 1 and make it ready to drive.
wbu_driver_set_throttle(0.0);
wbu_driver_set_gear(1);
wbu_driver_set_brake_intensity(0.0); %Brake intensity (brake pedal) is between 0 and 1.

my_gps = wb_robot_get_device('gps');
wb_gps_enable(my_gps, 10);

my_cam = wb_robot_get_device('camera');
wb_camera_enable(my_cam, 50);

%object_detection_nn = vgg16;
%figure;
classifier_obj = start_object_detector();
while wbu_driver_step() ~= -1
    disp('-----------------------------------');
    cur_speed = wbu_driver_get_current_speed();  % Reading current speed in km/h
    speed_error = TARGET_SPEED - cur_speed;
    target_control = speed_error * CONTROL_P_GAIN;
    target_control = min(1.0, max(-1.0, target_control));
    if target_control > 0  % We are slower then target speed.
        wbu_driver_set_throttle(target_control); %Throttle angle (gas pedal) is between 0 and 1.
        wbu_driver_set_brake_intensity(0.0); %Brake intensity (brake pedal) is between 0 and 1.
    else  % We are faster than target speed.
        wbu_driver_set_brake_intensity(-1*target_control);  %Brake intensity (brake pedal) is between 0 and 1.
        wbu_driver_set_throttle(0.0); %Throttle angle (gas pedal) is between 0 and 1.
    end
    
    % Apply controls:
    wbu_driver_set_steering_angle(0.0); % Positive values: steering to right. Negative values steering to left.
    %wbu_driver_set_throttle(1.0); %Throttle angle (gas pedal) is between 0 and 1.
    
    % Read current values:
    current_steering = wbu_driver_get_steering_angle(); % Only for reading the current steering. Not for controlling it.
    cur_throttle = wbu_driver_get_throttle(); % Only for reading the current throttle position. Not for controlling it.
    
    cur_image = wb_camera_get_image(my_cam);
    size(cur_image)
    [ det_boxes, det_probs, det_classes ] = object_detection(classifier_obj, cur_image);
    for ii = 1:length(det_classes)
        if det_classes(ii) == 0
            disp(['CAR at ', num2str(det_boxes(ii,:))]);
        elseif det_classes(ii) == 1
            disp(['PEDESTRIAN at ', num2str(det_boxes(ii,:))]);
        else
            disp(['CYCLIST at ', num2str(det_boxes(ii,:))]);
        end
    end
    image(cur_image);
    drawnow;
    %label = classify(object_detection_nn, cur_image);
    %label
    
    %imshow(cur_image) 
    %text(10, 20, char(label),'Color','white')
    %drawnow;
    
    
    cur_gear = wbu_driver_get_gear();  % Reading current gear.
    cur_position = wb_gps_get_values(my_gps);
    cur_gps_speed = wb_gps_get_speed(my_gps);
    
    disp(['GPS Speed: ', num2str(cur_gps_speed)]);
    disp(['GPS Position: ', num2str(cur_position)]);
    % Console outputs for debugging:
    disp(['cur gear: ', num2str(cur_gear)]);
    disp(['cur_throttle: ', num2str(cur_throttle)]);
    disp(['cur spd: ', num2str(cur_speed)]);
    
    %disp(['First 10 bytes of image: ', num2str(cur_image(1:10))]);
end