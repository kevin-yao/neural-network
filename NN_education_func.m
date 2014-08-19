function NN_education_func(trainingFile, testingFile)
learningRate = 0.05;
maxIterationNum = 1000;
epsilon = 0.001;
minError = 100000000;
numHiddenNeuron = 5;
[trainX,trainY,testX]=dataPreprocess(trainingFile, testingFile);
%for i =1: 1
outputHidden = zeros(size(trainX,1), numHiddenNeuron);
iterationCounter = 0;
%initialW1 = 0.01*rand(size(trainX, 2), numHiddenNeuron);
%w1=initialW1;
%initialW2 = 0.01*rand(numHiddenNeuron+1, 1);
%w2=initialW2;
w1=[0.0003    0.0035    0.0009    0.0049    0.0046;
    0.0021    0.0078    0.0059    0.0022    0.0064;
    0.0045    0.0044    0.0024    0.0023    0.0092;
    0.0013    0.0044    0.0084    0.0054    0.0016;
    0.0001    0.0005    0.0086    0.0076    0.0072;
    0.0073    0.0005    0.0096    0.0035    0.0058];

w2=[0.0043; 0.0088; 0.0039; 0.0018; 0.0063; 0.0062];
preError = 100000000;
while(1)
    iterationCounter = iterationCounter+1;
    % compute error
    outputHidden = sigmoid(trainX*w1);
    output = sigmoid([ones(size(outputHidden, 1), 1) outputHidden]*w2);
    error = sum((trainY-output).^2)/2;
    if mod(iterationCounter, 10)==0
        %          if error>preError
        %              break;
        %          end
        fprintf('%.1f\n', error);
        %        preError = error;
    end
    % test iteration condition
    if error < epsilon || iterationCounter > maxIterationNum
        break;
    end
    
    deltaOutput = zeros(size(trainX,1),1);
    deltaOutput = output.*(1-output).*(trainY-output);
    deltaHidden = zeros(size(trainX,1), numHiddenNeuron);
    w2Interim = w2(2:numHiddenNeuron+1, :);
    for i = 1: size(trainX,1)
        deltaHidden(i,:) = outputHidden(i,:).*(1-outputHidden(i,:)).*(w2Interim'.*deltaOutput(i));
    end
    
    % compute delta w1 and delta w2
    outputHidden = [ones(size(trainX, 1),1) outputHidden];
    deltaW2 = zeros(size(w2, 1), size(w2, 2));
    for i = 1: size(trainX, 1)
        deltaW2 =deltaW2 + (learningRate.*outputHidden(i,:).*deltaOutput(i))';
    end
    
    deltaW1 = zeros(size(w1, 1), size(w1, 2));
    for i = 1: size(deltaHidden , 1)
        for j = 1: size(deltaHidden, 2)
            deltaW1(:,j)= deltaW1(:,j)+(learningRate*deltaHidden(i, j).*trainX(i,:))';
        end
    end
    w1 = w1 + deltaW1;
    w2 = w2 + deltaW2;
end
fprintf('TRAINING COMPLETED! NOW PREDICTING.\n');
prediction = zeros(size(testX,1),1);
prediction = sigmoid([ones(size(testX,1),1) sigmoid(testX*w1)]*w2);
prediction = prediction.*100;
%target = load('education_dev_keys.txt');
%dataSize = size(target,1);
for i=1:size(testX,1)
    fprintf('%.1f\n', prediction(i));
end
%predictionError = sum((target-prediction).^2)/2

%if predictionError < minError
%    minError = predictionError;
%    bestW1 = initialW1;
%    bestW2 = initialW2;
%end
%end
end

function [trainX,trainY,testX]=dataPreprocess(trainFilePath, testFilePath)
data = textread(trainFilePath, '%s');
data = data(2:size(data,1),:);
sampleSize = size(data,1);
numAttribute = 5;
trainX = zeros(sampleSize,numAttribute);
trainY = zeros(sampleSize, 1);
for i=1:sampleSize
    for j=1:numAttribute+1
        line = data(i);
        line = line{1};
        line=strsplit(line,',');
        if j<= numAttribute
            trainX(i,j)=str2num(line{j});
        else
            trainY(i)=str2num(line{j});
        end
    end
end

data = textread(testFilePath, '%s');
data = data(2:size(data,1),:);
sampleSizeTest = size(data,1);
testX = zeros(sampleSizeTest,numAttribute);
for i=1:sampleSizeTest
    for j=1:numAttribute
        line = data(i);
        line = line{1};
        line=strsplit(line,',');
        testX(i,j)=str2num(line{j});
    end
end
maxValue = 100;
minValue = 0;
for i = 1:numAttribute
    trainX(:,i)=-1+((trainX(:,i)-minValue)./(maxValue-minValue)).*2;
    testX(:,i)=-1+((testX(:,i)-minValue)./(maxValue-minValue)).*2;
end
trainY = ((trainY(:,1)-minValue)./(maxValue-minValue)).*1;
trainX = [ones(sampleSize, 1) trainX];
testX = [ones(sampleSizeTest, 1) testX];
end

function y=sigmoid(x)
y=1.0./(1.0+exp(-x));
end
