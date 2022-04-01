%%-------------------------------------------------------------------------
% ���ߣ�       ������
% ���ڣ�       2021��6��
% ˵����       ����֪����������
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% ���ݴ��������������
clc
clear
close all

% ��������
addpath(genpath('data'))
table_female = importfile_train('FEMALE.TXT');
table_male = importfile_train('MALE.TXT');
table_test1 = importfile_test('test1.txt');
table_test2 = importfile_test('test2.txt');
table_test1.Gender(table_test1.Gender == 'f') = 'F';
table_test1.Gender(table_test1.Gender == 'm') = 'M';

% ��������ת�����������һ�����Ա�1��ʾŮ�ԣ�2��ʾ���ԣ�
train_set = [table_female.Height table_female.Weight ones(50, 1)];
train_set = [train_set; table_male.Height table_male.Weight 2*ones(50, 1)];
test1_set = [table_test1.Height table_test1.Weight];
test2_set = [table_test2.Height table_test2.Weight];
test1_set(table_test1.Gender == 'F', 3) = 1;
test1_set(table_test1.Gender == 'M', 3) = 2;
test2_set(table_test2.Gender == 'F', 3) = 1;
test2_set(table_test2.Gender == 'M', 3) = 2;

while 1
    k = input('ѡ����Լ����=');    %�ֶ�����ѡ����Լ����
    if k ~= 1 && k ~= 2 && k~=0
        disp('�Ƿ���ţ�����������')
    else
        break;
    end
end
if k == 0
    test_set = train_set;
else
    eval(['test_set = test' num2str(k) '_set;']);   %���ݵ�ǰѡ�еĲ��Լ�
end

C = size(train_set, 3);     %�����

while 1
    n_layers = input('�������=');    %�ֶ������������
    if mod(n_layers, 1) ~= 0 || n_layers <= 0
        disp('�Ƿ���ţ�����������')
    else
        break;
    end
end
while 1
    n_nodes = input('����ڵ����=');    %�ֶ���������ڵ����
    if mod(n_nodes, 1) ~= 0 || n_nodes <= 0
        disp('�Ƿ���ţ�����������')
    else
        break;
    end
end
X_train = train_set(:, 1:2)';
Y_train = train_set(:, 3)';
X_test = test_set(:, 1:2)';
Y_test = test_set(:, 3)';

%% ���֪��/������������
if n_layers == 1
    % Single Hidden Layer ������
    % Ĭ��ѵ���㷨, ѡ�����trainlm����
    net_SHL = feedforwardnet(n_nodes);  %����ڵ����
    net_SHL.trainParam.epochs = 50;  %��ѵ��epoch��
    % ѵ��������ģ�͡�
    [net_SHL,tr_SHL] = train(net_SHL, X_train, Y_train);

    % ����
    Y_pred = net_SHL(X_test);
    Y_pred = round(Y_pred);  %�������룬����[0.5, 2.5)�Ļع�ֵ
    Y_pred(Y_pred < 0.5) = 1;
    Y_pred(Y_pred >= 2.5) = 2;  %���������ع�ֵ
    perf = perform(net_SHL, Y_pred, Y_test);  %MSE  
    MSE = sum((Y_test-Y_pred).^2) / size(Y_test, 2);
    disp(['�������MSE��' num2str(MSE*100) '%'])
    view(net_SHL)
    plotperform(tr_SHL)
end
%% ������������
if n_layers ~= 1
    % Multiple Hidden Layer ������
    % Ĭ��ѵ���㷨, ѡ�����trainlm����
    n_layernodes = [];  %ÿ��ڵ�������
    for ii = 1:n_layers
        n_layernodes = [n_layernodes n_nodes];
    end
    net_MHL = feedforwardnet(n_layernodes);  %����ڵ����
    net_MHL.trainParam.epochs = 50;  %��ѵ��epoch��
    % ѵ��������ģ�͡�
    [net_MHL,tr_MHL] = train(net_MHL, X_train, Y_train);

    % ����
    Y_pred = net_MHL(X_test);
    Y_pred = round(Y_pred);  %�������룬����[0.5, 2.5)�Ļع�ֵ
    Y_pred(Y_pred < 0.5) = 1;
    Y_pred(Y_pred >= 2.5) = 2;  %���������ع�ֵ
    perf = perform(net_MHL, Y_pred, Y_test);  %MSE  
    MSE = sum((Y_test-Y_pred).^2) / size(Y_test, 2);
    disp(['�������MSE��' num2str(MSE*100) '%'])
    view(net_MHL)
    plotperform(tr_MHL)
end

%% ���ӻ�
figure
plot(X_train(1, Y_train==1), X_train(2, Y_train==1), 'ro'), hold on
plot(X_train(1, Y_train==2), X_train(2, Y_train==2), 'b*')
legend({'Female', 'Male'}, 'Location', 'best')
title('ѵ��������')
figure
plot(X_test(1, Y_test==1), X_test(2, Y_test==1), 'ro'), hold on
plot(X_test(1, Y_test==2), X_test(2, Y_test==2), 'b*')
legend({'Female', 'Male'}, 'Location', 'best')
title(['���Լ�' num2str(k) '����'])
figure
plot(X_test(1, Y_pred==1), X_test(2, Y_pred==1), 'ro'), hold on
plot(X_test(1, Y_pred==2), X_test(2, Y_pred==2), 'b*'), hold on
plot(X_test(1, Y_pred~=Y_test), X_test(2, Y_pred~=Y_test), 'ks', 'Markersize', 10), hold on
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['���Լ�' num2str(k) '��������������Ϊ' num2str(sum(Y_pred~=Y_test)/size(test_set, 1)*100) '%'])

%% �������ݺ�������ҳ���������ݣ���MATLAB�Զ����ɵĺ������룩
% % ѵ�������뺯��
function train_set = importfile_train(filename, startRow, endRow)
%IMPORTFILE ���ı��ļ��е���ֵ������Ϊ�����롣
%   TRAIN_SET = IMPORTFILE_TRAIN(FILENAME) ��ȡ�ı��ļ� FILENAME ��Ĭ��ѡ����Χ�����ݡ�
%
%   TRAIN_SET = IMPORTFILE_TRAIN(FILENAME, STARTROW, ENDROW) ��ȡ�ı��ļ� FILENAME ��
%   STARTROW �е� ENDROW ���е����ݡ�
%
% Example:
%   train_set = importfile_train('FEMALE.TXT', 1, 50);
%
%    ������� TEXTSCAN��

% �� MATLAB �Զ������� 2021/05/20 

% ��ʼ��������
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

% ����������Ϊ�ı���ȡ:
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%s%s%[^\n\r]';

% ���ı��ļ���
fileID = fopen(filename,'r');

% ���ݸ�ʽ��ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

% �ر��ı��ļ���
fclose(fileID);

% ��������ֵ�ı���������ת��Ϊ��ֵ��
% ������ֵ�ı��滻Ϊ NaN��
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2]
    % ������Ԫ�������е��ı�ת��Ϊ��ֵ���ѽ�����ֵ�ı��滻Ϊ NaN��
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % ����������ʽ�Լ�Ⲣɾ������ֵǰ׺�ͺ�׺��
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % �ڷ�ǧλλ���м�⵽���š�
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % ����ֵ�ı�ת��Ϊ��ֵ��
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end


% �����������
train_set = table;
train_set.Height = cell2mat(raw(:, 1));
train_set.Weight = cell2mat(raw(:, 2));
end

% % ���Լ����뺯��
function test_set = importfile_test(filename, startRow, endRow)
%IMPORTFILE ���ı��ļ��е���ֵ������Ϊ�����롣
%   TEST_SET = IMPORTFILE_TEST(FILENAME) ��ȡ�ı��ļ� FILENAME ��Ĭ��ѡ����Χ�����ݡ�
%
%   TEST_SET = IMPORTFILE_TEST(FILENAME, STARTROW, ENDROW) ��ȡ�ı��ļ� FILENAME ��
%   STARTROW �е� ENDROW ���е����ݡ�
%
% Example:
%   test_set = importfile_test('test1.txt', 1, 35);
%
%    ������� TEXTSCAN��

% �� MATLAB �Զ������� 2021/05/20 

% ��ʼ��������
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

% ÿ���ı��еĸ�ʽ:
%   ��1: ˫����ֵ (%f)
%	��2: ˫����ֵ (%f)
%   ��3: ���� (%C)
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%f%f%C%[^\n\r]';

% ���ı��ļ���
fileID = fopen(filename,'r');

% ���ݸ�ʽ��ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

% �ر��ı��ļ���
fclose(fileID);

% ���޷���������ݽ��еĺ���
% �ڵ��������δӦ���޷���������ݵĹ�����˲�����������롣Ҫ�����������޷���������ݵĴ��룬�����ļ���ѡ���޷������Ԫ����Ȼ���������ɽű���

% �����������
test_set = table(dataArray{1:end-1}, 'VariableNames', {'Height','Weight','Gender'});

end