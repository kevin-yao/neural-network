function NN_music_func( trainingFile, testingFile )
learningRate = 0.2;
maxIterationNum = 1000;
iterationCounter = 0;
epsilon = 0.001;
minErrorRate = 1;
numHiddenNeuron = 4;
[X,Y,testData]=dataPreprocess(trainingFile, testingFile);
%for i = 1:30
iterationCounter = 0;
outputHidden = zeros(size(X,1), numHiddenNeuron);
%initialW1 = 0.01*rand(size(X, 2), numHiddenNeuron);
%w1 = initialW1;
%initialW2 = 0.01*rand(numHiddenNeuron+1, 1);
%w2 = initialW2;
w1=[0.0010    0.0061    0.0094    0.0083;
    0.0055    0.0078    0.0087    0.0032;
    0.0040    0.0057    0.0051    0.0098;
    0.0011    0.0081    0.0079    0.0028;
    0.0072    0.0058    0.0047    0.0007];
w2=[0.0075; 0.0083; 0.0092; 0.0033; 0.0080];
preError = 100000000;
while(1)
    iterationCounter = iterationCounter+1;
    % compute error
    outputHidden = sigmoid(X*w1);
    output = sigmoid([ones(size(outputHidden, 1), 1) outputHidden]*w2);
    error = sum((Y-output).^2)/2;
    if mod(iterationCounter, 10)==0
        %  		if error>preError
        %              break;
        %          end
        fprintf('%.1f\n', error);
        %		preError = error;
    end
    % test iteration condition
    if error < epsilon || iterationCounter > maxIterationNum
        break;
    end
    
    deltaOutput = zeros(size(X,1),1);
    deltaOutput = output.*(1-output).*(Y-output);
    deltaHidden = zeros(size(X,1), numHiddenNeuron);
    w2Interim = w2(2:numHiddenNeuron+1, :);
    for i = 1: size(X,1)
        deltaHidden(i,:)=outputHidden(i,:).*(1-outputHidden(i,:)).*(w2Interim'.*deltaOutput(i));
    end
    % compute delta w1 and delta w2
    outputHidden = [ones(size(X, 1),1) outputHidden];
    deltaW2 = zeros(size(w2, 1), size(w2, 2));
    for i = 1: size(X, 1)
        deltaW2 =deltaW2+(learningRate.*outputHidden(i,:).*deltaOutput(i))';
    end
    
    deltaW1 = zeros(size(w1, 1), size(w1, 2));
    for i = 1: size(deltaHidden , 1)
        for j = 1: size(deltaHidden, 2)
            deltaW1(:,j)= deltaW1(:,j)+(learningRate*deltaHidden(i, j).*X(i,:))';
        end
    end
    w1 = w1 + deltaW1;
    w2 = w2 + deltaW2;
end
fprintf('TRAINING COMPLETED! NOW PREDICTING.\n');
prediction = zeros(size(testData,1),1);
prediction = sigmoid([ones(size(testData,1),1) sigmoid(testData*w1)]*w2);
%  target = textread('music_dev_keys.txt', '%s');
%  errorCounter = 0;
%  dataSize = size(target,1);
for i=1:size(prediction, 1)
    if prediction(i) >= 0.5
        fprintf('yes\n');
%                  if strcmpi(target(i), 'no')
%                      errorCounter=errorCounter+1;
%                  end
    else
        fprintf('no\n');
%                  if strcmpi(target(i), 'yes')
%                      errorCounter=errorCounter+1;
%                  end
    end
end
%errorRate = errorCounter/dataSize
% if errorRate < minErrorRate
%     minErrorRate = errorRate;
%     bestW1 = initialW1;
%     bestW2 = initialW2;
% end
%end
end

function [X,Y,testX]=dataPreprocess(trainFilePath, testFilePath)
data = textread(trainFilePath, '%s');
data = data(2:size(data,1),:);
sampleSize = size(data,1);
numAttribute = 4;
X = zeros(sampleSize,numAttribute);
Y = zeros(sampleSize, 1);
for i=1:sampleSize
    for j=1:numAttribute+1
        line = data(i);
        line = line{1};
        line=strsplit(line,',');
        if j<= 2
            X(i,j)=str2num(line{j});
        elseif j<= numAttribute
            if strcmpi(line{j}, 'yes')
                X(i,j) = 1;
            else
                X(i,j) = -1;
            end
        else
            if strcmpi(line{j}, 'yes')
                Y(i) = 1;
            else
                Y(i) = 0;
            end
        end
    end
end

data = textread(testFilePath, '%s');
data = data(2:size(data,1),:);
sampleSizeTest = size(data,1);
testX = zeros(sampleSizeTest,numAttribute);
for i=1:sampleSizeTest
    for j=1:4
        line = data(i);
        line = line{1};
        line=strsplit(line,',');
        if j<= 2
            testX(i,j)=str2num(line{j});
        elseif j<= numAttribute
            if strcmpi(line{j}, 'yes')
                testX(i,j) = 1;
            else
                testX(i,j) = -1;
            end
        end
    end
end
maxValue = 2000;
minValue = 1900;
X(:,1)=-1+((X(:,1)-(minValue))./(maxValue-minValue)).*2;
testX(:,1)=-1+((testX(:,1)-(minValue))./(maxValue-minValue)).*2;
maxValue = 7;
minValue = 0;
X(:,2)=-1+((X(:,2)-(minValue))./(maxValue-minValue)).*2;
testX(:,2)=-1+((testX(:,2)-(minValue))./(maxValue-minValue)).*2;
X = [ones(sampleSize, 1) X];
testX = [ones(sampleSizeTest, 1) testX];
end

function y=sigmoid(x)
y=1.0./(1.0+exp(-x));
end