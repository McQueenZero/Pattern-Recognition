%%-------------------------------------------------------------------------
% 作者：       赵敏琨
% 日期：       2021年6月
% 说明：       多层感知机、神经网络
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 数据处理、输入输出交互
clc
clear
close all

% 导入数据
addpath(genpath('data'))
table_female = importfile_train('FEMALE.TXT');
table_male = importfile_train('MALE.TXT');
table_test1 = importfile_test('test1.txt');
table_test2 = importfile_test('test2.txt');
table_test1.Gender(table_test1.Gender == 'f') = 'F';
table_test1.Gender(table_test1.Gender == 'm') = 'M';

% 数据类型转换，矩阵最后一列是性别（1表示女性，2表示男性）
train_set = [table_female.Height table_female.Weight ones(50, 1)];
train_set = [train_set; table_male.Height table_male.Weight 2*ones(50, 1)];
test1_set = [table_test1.Height table_test1.Weight];
test2_set = [table_test2.Height table_test2.Weight];
test1_set(table_test1.Gender == 'F', 3) = 1;
test1_set(table_test1.Gender == 'M', 3) = 2;
test2_set(table_test2.Gender == 'F', 3) = 1;
test2_set(table_test2.Gender == 'M', 3) = 2;

while 1
    k = input('选择测试集编号=');    %手动输入选择测试集编号
    if k ~= 1 && k ~= 2 && k~=0
        disp('非法编号，请重新输入')
    else
        break;
    end
end
if k == 0
    test_set = train_set;
else
    eval(['test_set = test' num2str(k) '_set;']);   %传递当前选中的测试集
end

C = size(train_set, 3);     %类别数

while 1
    n_layers = input('隐层个数=');    %手动输入隐层个数
    if mod(n_layers, 1) ~= 0 || n_layers <= 0
        disp('非法编号，请重新输入')
    else
        break;
    end
end
while 1
    n_nodes = input('隐层节点个数=');    %手动输入隐层节点个数
    if mod(n_nodes, 1) ~= 0 || n_nodes <= 0
        disp('非法编号，请重新输入')
    else
        break;
    end
end
X_train = train_set(:, 1:2)';
Y_train = train_set(:, 3)';
X_test = test_set(:, 1:2)';
Y_test = test_set(:, 3)';

%% 多感知机/单隐层神经网络
if n_layers == 1
    % Single Hidden Layer 单隐层
    % 默认训练算法, 选择的是trainlm函数
    net_SHL = feedforwardnet(n_nodes);  %隐层节点个数
    net_SHL.trainParam.epochs = 50;  %总训练epoch数
    % 训练神经网络模型。
    [net_SHL,tr_SHL] = train(net_SHL, X_train, Y_train);

    % 测试
    Y_pred = net_SHL(X_test);
    Y_pred = round(Y_pred);  %四舍五入，修正[0.5, 2.5)的回归值
    Y_pred(Y_pred < 0.5) = 1;
    Y_pred(Y_pred >= 2.5) = 2;  %修正其他回归值
    perf = perform(net_SHL, Y_pred, Y_test);  %MSE  
    MSE = sum((Y_test-Y_pred).^2) / size(Y_test, 2);
    disp(['均方误差MSE：' num2str(MSE*100) '%'])
    view(net_SHL)
    plotperform(tr_SHL)
end
%% 多隐层神经网络
if n_layers ~= 1
    % Multiple Hidden Layer 多隐层
    % 默认训练算法, 选择的是trainlm函数
    n_layernodes = [];  %每层节点数向量
    for ii = 1:n_layers
        n_layernodes = [n_layernodes n_nodes];
    end
    net_MHL = feedforwardnet(n_layernodes);  %隐层节点个数
    net_MHL.trainParam.epochs = 50;  %总训练epoch数
    % 训练神经网络模型。
    [net_MHL,tr_MHL] = train(net_MHL, X_train, Y_train);

    % 测试
    Y_pred = net_MHL(X_test);
    Y_pred = round(Y_pred);  %四舍五入，修正[0.5, 2.5)的回归值
    Y_pred(Y_pred < 0.5) = 1;
    Y_pred(Y_pred >= 2.5) = 2;  %修正其他回归值
    perf = perform(net_MHL, Y_pred, Y_test);  %MSE  
    MSE = sum((Y_test-Y_pred).^2) / size(Y_test, 2);
    disp(['均方误差MSE：' num2str(MSE*100) '%'])
    view(net_MHL)
    plotperform(tr_MHL)
end

%% 可视化
figure
plot(X_train(1, Y_train==1), X_train(2, Y_train==1), 'ro'), hold on
plot(X_train(1, Y_train==2), X_train(2, Y_train==2), 'b*')
legend({'Female', 'Male'}, 'Location', 'best')
title('训练集数据')
figure
plot(X_test(1, Y_test==1), X_test(2, Y_test==1), 'ro'), hold on
plot(X_test(1, Y_test==2), X_test(2, Y_test==2), 'b*')
legend({'Female', 'Male'}, 'Location', 'best')
title(['测试集' num2str(k) '数据'])
figure
plot(X_test(1, Y_pred==1), X_test(2, Y_pred==1), 'ro'), hold on
plot(X_test(1, Y_pred==2), X_test(2, Y_pred==2), 'b*'), hold on
plot(X_test(1, Y_pred~=Y_test), X_test(2, Y_pred~=Y_test), 'ks', 'Markersize', 10), hold on
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['测试集' num2str(k) '分类结果：错误率为' num2str(sum(Y_pred~=Y_test)/size(test_set, 1)*100) '%'])

%% 导入数据函数（主页→导入数据，由MATLAB自动生成的函数代码）
% % 训练集导入函数
function train_set = importfile_train(filename, startRow, endRow)
%IMPORTFILE 将文本文件中的数值数据作为矩阵导入。
%   TRAIN_SET = IMPORTFILE_TRAIN(FILENAME) 读取文本文件 FILENAME 中默认选定范围的数据。
%
%   TRAIN_SET = IMPORTFILE_TRAIN(FILENAME, STARTROW, ENDROW) 读取文本文件 FILENAME 的
%   STARTROW 行到 ENDROW 行中的数据。
%
% Example:
%   train_set = importfile_train('FEMALE.TXT', 1, 50);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2021/05/20 

% 初始化变量。
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

% 将数据列作为文本读取:
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%s%s%[^\n\r]';

% 打开文本文件。
fileID = fopen(filename,'r');

% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

% 关闭文本文件。
fclose(fileID);

% 将包含数值文本的列内容转换为数值。
% 将非数值文本替换为 NaN。
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2]
    % 将输入元胞数组中的文本转换为数值。已将非数值文本替换为 NaN。
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % 创建正则表达式以检测并删除非数值前缀和后缀。
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % 在非千位位置中检测到逗号。
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % 将数值文本转换为数值。
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


% 创建输出变量
train_set = table;
train_set.Height = cell2mat(raw(:, 1));
train_set.Weight = cell2mat(raw(:, 2));
end

% % 测试集导入函数
function test_set = importfile_test(filename, startRow, endRow)
%IMPORTFILE 将文本文件中的数值数据作为矩阵导入。
%   TEST_SET = IMPORTFILE_TEST(FILENAME) 读取文本文件 FILENAME 中默认选定范围的数据。
%
%   TEST_SET = IMPORTFILE_TEST(FILENAME, STARTROW, ENDROW) 读取文本文件 FILENAME 的
%   STARTROW 行到 ENDROW 行中的数据。
%
% Example:
%   test_set = importfile_test('test1.txt', 1, 35);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2021/05/20 

% 初始化变量。
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

% 每个文本行的格式:
%   列1: 双精度值 (%f)
%	列2: 双精度值 (%f)
%   列3: 分类 (%C)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%f%f%C%[^\n\r]';

% 打开文本文件。
fileID = fopen(filename,'r');

% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

% 关闭文本文件。
fclose(fileID);

% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

% 创建输出变量
test_set = table(dataArray{1:end-1}, 'VariableNames', {'Height','Weight','Gender'});

end