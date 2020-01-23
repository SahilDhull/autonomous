%% Core_py2matlab
%
% How to configure Python in MATLAB 2015b (on a Windows Box):
%
% # Download Python 3.4 from <python.org/downloads/>. Ensure that the
% 32-bit or 64-bit version of Python is consistent with MATLAB.
% # Update the Windows Path Environment Variable to include the Python path
% and the _PythonXX\Scripts_ path.
%
% Copyright (c) 2015, Kyle 
% Copyright (c) 2015, Kyle Wayne Karhohs 
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
% * Redistributions of source code must retain the above copyright 
% notice, this list of conditions and the following disclaimer. 
% * Redistributions in binary form must reproduce the above copyright 
% notice, this list of conditions and the following disclaimer in 
% the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
%
% Downloaded from Mathworks File Exchange
%
% Notes: 
%
% # Data structures that are more than 2 dimensions are not handled.
% # Data structures must be fully defined, i.e. no empty brackets [].
function matlabData = Core_py2matlab(pythonData)
%%%
% convert Python into MATLAB
matlabData = recursiveFunPy2Matlab(pythonData);
%%%
% Matrices of numbers are converted into cells, so another function will
% reformat these into matrices.
matlabData = recursiveFunCell2MatCheck(matlabData);
%% pythonConversion
% <matlab:doc('handling-data-returned-from-python') data from Python>
    function matlabData = pythonConversion(pyData)
        pyType = class(pyData);
        switch pyType
            case {'py.str','py.unicode'}
                matlabData = char(pyData);
            case 'py.bytes'
                matlabData = uint8(pyData);
            case {'py.int','py.long','py.array.array'}
                matlabData = double(pyData);
            case {'py.list','py.tuple'}
                matlabData = cell(pyData);
            case 'py.dict'
                matlabData = struct(pyData);
            otherwise
                matlabData = pyData;
        end
    end
%% recursiveFunPy2Matlab
% Loops through the Python data types and converts them into MATLAB data
% types
    function matlabData = recursiveFunPy2Matlab(pyData)
        matlabData = pythonConversion(pyData);
        matlabType = class(matlabData);
        mynum = numel(matlabData);
        switch matlabType
            case 'cell'
                for i = 1:mynum
                    matlabData{i} = recursiveFunPy2Matlab(matlabData{i});
                end
            case 'struct'
                for i = 1:mynum
                    myfields = fieldnames(matlabData(i));
                    for j = 1:numel(myfields)
                        matlabData(i).(myfields{j}) = recursiveFunPy2Matlab(matlabData(i).(myfields{j}));
                    end
                end
        end
    end
%% recursiveFunCell2MatCheck
% A second loop through the data structure identifies numeric matrices
% stored as cells
    function mydata = recursiveFunCell2MatCheck(mydata)
        myType = class(mydata);
        mynum = numel(mydata);
        switch myType
            case 'cell'
                if iscellstr(mydata)
                    return
                elseif all(cellfun(@isnumeric,mydata(:)))
                    % This is the key condition. A cell full of numbers
                    % will be converted into a matrix.
                    try
                        mydata = cell2mat(mydata);
                    catch
                        %do nothing
                    end
                else
                    for i = 1:mynum
                        mydata{i} = recursiveFunCell2MatCheck(mydata{i});
                    end
                    if all(cellfun(@isnumeric,mydata(:)))
                        % This is to handle 2D matrices. Anything that has
                        % more than 2 dimensions will be flattened to 2D.
                        try
                            mydata = cell2mat(transpose(mydata));
                        catch
                            %do nothing
                        end
                    end
                end
            case 'struct'
                for i = 1:numel(mydata)
                    myfields = fieldnames(mydata(i));
                    for j = 1:numel(myfields)
                        mydata(i).(myfields{j}) = recursiveFunCell2MatCheck(mydata(i).(myfields{j}));
                    end
                end
        end
    end
end