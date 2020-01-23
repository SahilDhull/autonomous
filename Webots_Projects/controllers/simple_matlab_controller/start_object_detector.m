function [ classifier_obj ] = start_object_detector( )
%start_object_detector Summary of this function goes here
%   Detailed explanation goes here
% Import Python modules:
try
    pyversion path_to\python.exe
catch
    pyversion
end
insert(py.sys.path,int32(0),'D:/PerceptionVnV');
insert(py.sys.path,int32(0),'D:/PerceptionVnV/Sim_ATAV/classifier/classifier_interface');

classifier_obj = py.classifier.Classifier();
classifier_obj.start_classification_engine();
end

