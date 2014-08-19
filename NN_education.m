arg_list=argv();
trainingFilePath = arg_list{1};
testingFilePath = arg_list{2};
NN_education_func(trainingFilePath, testingFilePath);