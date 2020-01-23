@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
ECHO **********************************************************************
ECHO **** This will guide you to install required packages for SimATAV ****
ECHO **** ------------------------------------------------------------ ****
ECHO **** Depending on your setup, you may need administrative rights. ****
ECHO **** ------------------------------------------------------------ ****
ECHO ****                     Press Any Key to Continue                ****
ECHO **********************************************************************
pause
ECHO Please download Python_Dependencies from http://www.public.asu.edu/~etuncali/downloads/ and unzip it next to this installation script.
ECHO You should have whl files directly under ./Python_Dependencies/
ECHO Press any key to continue after you are done downloading and unzipping Python_Dependencies.
pause
SET /P isproxy="Do you want to apply proxy settings? [y/n]>"
IF "%isproxy%"=="Y" (SET isproxy = "y")
IF "%isproxy%"=="y" (
	SET /p proxy_setting="Please enter your proxy setting: (example: http://user_name@companyproxy.com:portno) > "
	SET proxy_str="--proxy=!proxy_setting!"
	ECHO I will use the following proxyoption with pip3: !proxy_str!
) ELSE (
	ECHO Not using Proxy.
	SET proxy_str=" "
)

SET /P ispythoninstalled="Do you have already have Python 3.7 64 Bit installed? To try, enter 'python' in command prompt and check the version. [y/n]>"
IF "%ispythoninstalled%"=="Y" (SET ispythoninstalled = "y")
IF "%ispythoninstalled%"=="y" (
	ECHO Python OK.
) ELSE (
	ECHO Please install python 3.7 64 Bit, and add to the system path, from https://www.python.org/downloads/release/python-371/
	ECHO Press any key to continue after you are done installing Python.
	pause
)
SET /P iswebotsinstalled="Do you have already have Webots r2019a installed? [y/n]>"
IF "%iswebotsinstalled%"=="Y" (SET iswebotsinstalled = "y")
IF "%iswebotsinstalled%"=="y" (
	ECHO Webots OK.
) ELSE (
	ECHO Please install Webots r2019a from https://www.cyberbotics.com.
	ECHO Press any key to continue after you are done installing Webots.
	pause
)
SET /P isfalsification="Do you want to try robustness-guided falsification (you will need to Matlab and S-TaLiRo)? [y/n]>"
IF "%isfalsification%"=="Y" (SET isfalsification = "y")
IF "%isfalsification%"=="y" (
	SET /P ismatlabinstalled="Do you have already have Matlab installed (Sim-ATAV is tested with Matlab r2017b)? [y/n]>"
	IF "%ismatlabinstalled%"=="Y" (SET ismatlabinstalled = "y")
	IF "%ismatlabinstalled%"=="y" (
		ECHO Matlab OK.
	) ELSE (
		ECHO Please install Matlab, Sim-ATAV is tested with Matlab r2017b, from http://www.mathworks.com/ if you want to try robustness-guided falsification with S-TaLiRo.
		ECHO Press any key to continue after you are done installing Matlab.
		pause
	)
	ECHO Please install S-TaLiRo from https://sites.google.com/a/asu.edu/s-taliro/s-taliro
	ECHO Press any key to continue after you are done installing S-TaLiRo.
	pause
) ELSE (
	ECHO Not installing Matlab or S-TaLiRo.
)
SET /P iscovering="Do you want to design your Covering Array based tests (you will need to ACTS from NIST)? [y/n]>"
IF "%iscovering%"=="Y" (SET iscovering = "y")
IF "%iscovering%"=="y" (
	ECHO Please request a copy of ACTS from NIST at https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools#acts
	ECHO You will need to wait for an e-mail to install ACTS, this may take a couple of days.
	ECHO Press any key to continue after you are done requesting ACTS. For now we will continue with the Sim-ATAV installation.
	pause
) ELSE (
	ECHO Not installing ACTS.
)
SET /P isgpu="Do you have a CUDA-enabled GPU? [y/n]>"
IF "%isgpu%"=="Y" (SET isgpu = "y")
ECHO %isgpu%. 
IF "%isgpu%"=="y" (
	ECHO Has GPU.
) ELSE (
	ECHO No GPU.
)

ECHO Processors with AVX2 support can run tensorflow faster.
ECHO A list of CPUs with AVX2 is here: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2
SET /P isavx="Does your processor support AVX2? [y/n]>"
IF "%isavx%"=="Y" (SET isavx = "y")
ECHO %isavx%. 
IF "%isavx%"=="y" (
	ECHO CPU with AVX2.
) ELSE (
	ECHO CPU without AVX2.
)

IF "%isgpu%"=="y" (
ECHO With a CUDA-enabled GPU, you can have increased performance by installing CUDA-Toolkit 9.0 and CUDNN 7.0.
SET /P iscudainstalled="Do you already have CUDA-Toolkit 10.0 and CUDNN 7.3.1 installed? [y/n]>"
IF "%iscudainstalled%"=="Y" (SET iscudainstalled = "y")
ECHO %iscudainstalled%. 
IF "%iscudainstalled%"=="y" (
	ECHO CUDA Installed.
) ELSE (
	ECHO CUDA Not installed yet.
)
IF "%iscudainstalled%"=="y" (
	ECHO After the installation, or now, please go to Sim_ATAV/classifier/classifier_interface/gpu_check.py file and set has_gpu to True.
	ECHO Press any key to continue installation.
	pause
) ELSE (
SET /P willinstallcuda="Do you want to install CUDA-Toolkit 10.0 and CUDNN 7.3.1 for increased performance? [y/n]>"
IF "%willinstallcuda%"=="Y" (SET willinstallcuda = "y")
ECHO %willinstallcuda%. 
IF "%willinstallcuda%"=="y" (
	ECHO Will install CUDA.
) ELSE (
	ECHO Will not install CUDA.
)
IF "%willinstallcuda%"=="y" (
	ECHO Please install CUDA-Toolkit 10.0 from http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
	ECHO and CUDNN 7.3.1 from https://developer.nvidia.com/rdp/cudnn-download.
	ECHO After the installation, or now, please go to Sim_ATAV/classifier/classifier_interface/gpu_check.py file and set has_gpu to True.
	ECHO Press any key to continue after you are done installing CUDA-Toolkit and CUDNN.
) ELSE (
	SET isgpu = "n"
	ECHO You will not take advantage of your CUDA-enabled GPU. Press any key to continue.
)
pause
)
)
ECHO The installer will now try to automatically install python dependencies.

ECHO **** Installing Numpy+MKL ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user Python_Dependencies\numpy-1.14.6+mkl-cp37-cp37m-win_amd64.whl --upgrade
) ELSE (
	pip3 install --user Python_Dependencies\numpy-1.14.6+mkl-cp37-cp37m-win_amd64.whl --upgrade
)

ECHO **** Installing Scipy ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user -Iv scipy==1.2.0
) ELSE (
	pip3 install --user -Iv scipy==1.2.0
)

ECHO **** Installing scikit-learn ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user scikit-learn
) ELSE (
	pip3 install --user scikit-learn
)

ECHO **** Installing OpenCV+Contrib ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user -Iv opencv-contrib-python>=3.4.0
) ELSE (
	pip3 install --user -Iv opencv-contrib-python>=3.4.0
)

ECHO **** Installing Absl Py ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user absl-py
) ELSE (
	pip3 install --user absl-py
)

ECHO **** Installing matplotlib ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user matplotlib
) ELSE (
	pip3 install --user matplotlib
)

IF "%isgpu%"=="y" (
ECHO **** Installing TensorFlow for GPU ****
IF "%isproxy%"=="y" (
IF "%isavx%"=="y" (
	ECHO **** Installing with AVX2 Support ****
	pip3 install %proxy_str% --user --upgrade Python_Dependencies\tensorflow_gpu-1.12.0-cp37-cp37m-win_amd64.whl
) ELSE (
	ECHO **** Installing without AVX2 Support ****
	pip3 install %proxy_str% --user --upgrade Python_Dependencies\sse2\tensorflow_gpu-1.12.0-cp37-cp37m-win_amd64.whl
)
) ELSE (
IF "%isavx%"=="y" (
	ECHO **** Installing with AVX2 Support ****
	pip3 install --user --upgrade Python_Dependencies\tensorflow_gpu-1.12.0-cp37-cp37m-win_amd64.whl
) ELSE (
	ECHO **** Installing without AVX2 Support ****
	pip3 install --user --upgrade Python_Dependencies\sse2\tensorflow_gpu-1.12.0-cp37-cp37m-win_amd64.whl
)
)
) ELSE (
ECHO **** Installing TensorFlow without GPU support****
IF "%isproxy%"=="y" (
IF "%isavx%"=="y" (
	ECHO **** Installing with AVX2 Support ****
	pip3 install %proxy_str% --user --upgrade Python_Dependencies\tensorflow-1.12.0-cp37-cp37m-win_amd64.whl
) ELSE (
	ECHO **** Installing without AVX2 Support ****
	pip3 install %proxy_str% --user --upgrade Python_Dependencies\sse2\tensorflow-1.12.0-cp37-cp37m-win_amd64.whl
)
) ELSE (
IF "%isavx%"=="y" (
	ECHO **** Installing with AVX2 Support ****
	pip3 install --user --upgrade Python_Dependencies\tensorflow-1.12.0-cp37-cp37m-win_amd64.whl
) ELSE (
	ECHO **** Installing without AVX2 Support ****
	pip3 install --user --upgrade Python_Dependencies\sse2\tensorflow-1.12.0-cp37-cp37m-win_amd64.whl
)
)
)

ECHO **** Installing Pandas ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user pandas
) ELSE (
	pip3 install --user pandas
)

ECHO **** Installing Pillow ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user Pillow
) ELSE (
	pip3 install --user Pillow
)

ECHO **** Installing Joblib ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user joblib
) ELSE (
	pip3 install --user joblib
)

ECHO **** Installing Shapely which is used in path tracking controllers ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user Python_Dependencies\Shapely-1.6.4.post1-cp37-cp37m-win_amd64.whl
) ELSE (
	pip3 install --user Python_Dependencies\Shapely-1.6.4.post1-cp37-cp37m-win_amd64.whl
)

ECHO **** Installing pykalman ****
IF "%isproxy%"=="y" (
	pip3 install %proxy_str% --user pykalman
) ELSE (
	pip3 install --user pykalman
)

ECHO **** Installing Easydict ****
cd Python_Dependencies\easydict-1.7\
python setup.py install
cd ..\..

ECHO **** Installing pydubins ****
cd Python_Dependencies\pydubins-master
python setup.py install
cd ..\..

ECHO **** Please check any error mesages above. You can close this window! ****
IF "%isgpu%"=="y" (
	ECHO If you still haven't done this, please go to Sim_ATAV/classifier/classifier_interface/gpu_check.py file and set has_gpu to True to utilize you CUDA-enabled GPU.
)
ECHO ! You can delete Python package wheel files that are under Python_Dependencies folder to save some disk space.
ECHO Press any key to exit.
pause

