global FALSIF_EXP_RESULTS_FILE_NAME
global FALSIF_Run_Number
global Falsif_Test_Type

TOTAL_FALSIF_RESTARTS = 1;
TOTAL_SA_BUDGET = 500;
SINGLE_SA_BUDGET = 100;

for FALSIF_Run_Number = 1:TOTAL_FALSIF_RESTARTS
    FALSIF_Run_Number
   
    %-----------------------------
    disp('Simulated Annealing...')
    FALSIFICATION_SOLVER = 'SA_Taliro';
    Falsif_Test_Type = 'SA';
    FALSIF_EXP_RESULTS_FILE_NAME = [FALSIF_EXP_RESULTS_FILE_PREFIX, FALSIFICATION_SOLVER, '_', num2str(FALSIF_Run_Number), '.csv'];
    success = false;
    while(~success)
        try
            run_staliro_after_ca_test1_r1;
            success = true;
        catch
            disp('again')
            pause(15)
        end
    end
    save([MATLAB_LOGS_FOLDER, FALSIFICATION_SOLVER, '_Run_after_CA_',num2str(FALSIF_Run_Number),'.mat']);
    
%     %-----------------------------
%     disp('Uniform Random...')
%     FALSIFICATION_SOLVER = 'UR_Taliro';
%     Falsif_Test_Type = 'UR';
%     FALSIF_EXP_RESULTS_FILE_NAME = [FALSIF_EXP_RESULTS_FILE_PREFIX, FALSIFICATION_SOLVER, '_', num2str(FALSIF_Run_Number), '.csv'];
%     success = false;
%     while(~success)
%         try
%             run_staliro_after_ca_test1_r1;
%             success = true;
%         catch
%             disp('again')
%             pause(15)
%         end
%     end
%     save([MATLAB_LOGS_FOLDER, FALSIFICATION_SOLVER, '_Run_after_CA_',num2str(FALSIF_Run_Number),'.mat']);
end
disp('Finished!!!');
