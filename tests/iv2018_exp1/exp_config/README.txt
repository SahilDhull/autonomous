The .csv files here are generated by the ACTS covering array generation tool provided by NIST.
See: https://csrc.nist.gov/Projects/Automated-Combinatorial-Testing-for-Software

File name convention:
$EXPERIMENT_NAME$_CA.xml : ACTS test project description for test scenario names as $EXPERIMENT_NAME$. Contains the parameter names, values etc.
$EXPERIMENT_NAME$_CA_$n$_way_$extra$.csv : The ACTS output covering array information. Covering array covers $n$-way combinations. $extra$ is optional, and it lists additional information when applicable. The csv file contains a header with the covering array generation setup. In its body, it contains the generated covering array toget with the parameter names as column titles.
$EXPERIMENT_NAME$_CA_$n$_way_TEST_RESULTS.csv : Contains test results (for instance robustness values) for each test.
