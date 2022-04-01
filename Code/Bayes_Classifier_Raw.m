%%-------------------------------------------------------------------------
% 作者：       赵敏琨         学号：2018302068     
% 日期：       2021年5月
% 说明：       贝叶斯分类器编程题：用身高和/或体重数据进行性别分类的实验
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 主文件：数据处理、输入输出交互
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

% 训练集数据类型转换，矩阵第一页是女性，矩阵第二页是男性
train_set(:,:,1) = [table_female.Height table_female.Weight];
train_set(:,:,2) = [table_male.Height table_male.Weight];

% 测试集数据类型转换，矩阵最后一列是性别（1表示女性，2表示男性）
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
while 1
    P_pr(1,1) = input('第一类(女)先验概率P_pr(1)=');    %手动输入先验概率
    P_pr(1,2) = 1 - P_pr(1,1);
    if 0 <= P_pr(1,1) && P_pr(1,1) <= 1
        break;
    else
        disp('非法先验概率，请重新输入')
    end
end

if k == 0
    test_set = [train_set(:,:,1); train_set(:,:,2)];    %应用到训练集上
else
    eval(['test_set = test' num2str(k) '_set;']);   %传递当前选中的测试集
end

while 1
    feature = input('采用的特征为（H/W/B）：', 's');    %手动输入采用的特征
    %Height/Weight/Both
    switch feature
        case 'H'
            if k ~= 0
                train_set(:,2,:) = [];
                test_set(:,2) = [];
                break;
            else
                disp('非法，请重新输入')
            end
        case 'W'
            if k ~= 0
                train_set(:,1,:) = [];
                test_set(:,1) = [];
                break;
            else
                disp('非法，请重新输入')
            end
        case 'B'
            % 不做删除处理
            while 1
                relevant = input('特征是否相关（Y/N）：', 's');   %手动输入特征是否相关
                if relevant == 'N' || relevant == 'n'
                    flag = 0;
                    break;
                elseif relevant == 'Y' || relevant == 'y'
                    flag = 1;
                    break;
                else
                    disp('非法，请重新输入')
                end
            end           
            break;
        otherwise
            disp('非法，请重新输入')
    end
end

while 1
    Method = input('决策准则（最小错误率/最小风险）(E/R)：', 's');    %手动选择方法
    %Error/Risk
    switch Method 
        case 'E'
            break;
        case 'R'
            break;
        otherwise
            disp('非法，请重新输入')
    end
end

C = size(train_set, 3);     %类别数

% 最小错误率Bayes分类器
if Method == 'E'
    [P_po, fg_array] = Gauss_BayesClassifier(train_set, test_set, P_pr, flag);  %返回后验概率和决策方程组
    % P_po每列代表一类（1表示女f，2表示男m）,每行代表一个样本
    % 每列找最大值，返回列索引，1说明女性后验概率大，2说明男性后验概率大
    [~, I] = max(P_po, [], C); 
end

% 最小风险Bayes分类器
% 决策表为：
% 后验概率  女  男
% 判断为女 0   1.2
% 判断为男 0.8   0
if Method == 'R'
    [P_po, fg_array] = Gauss_BayesClassifier(train_set, test_set, P_pr, flag);  %返回后验概率和决策方程组
    LBD = [0 1.2; 0.8 0]';
    P_R = P_po * LBD;
    fg_array = log(exp(fg_array) * LBD);
    [~, I] = min(P_R, [], C);

end

fg = fg_array(2) - fg_array(1);     %决策面方程
    
% 写入测试集表格方便对照
if k == 0
    table_train = table();
    table_train.Height = test_set(:,1);
    table_train.Weight = test_set(:,2);
    table_train.Gender(1:50) = 'F';
    table_train.Gender(51:100) = 'M';
    table_train.Classified(I == 1) = 'F';
    table_train.Classified(I == 2) = 'M';
    % disp(table_train)
else
    eval(['table_test' num2str(k) '.Classified(I == 1) = ''F'';']);
    eval(['table_test' num2str(k) '.Classified(I == 2) = ''M'';']);
    % eval(['disp(table_test' num2str(k) ')'])
end

% 计算错误率
N = size(test_set, 1);
Wc = 0;     %分类错误计数
Wi = [];    %分类错误索引
for m = 1:N
    if k == 0
        if table_train.Gender(m) ~= table_train.Classified(m)
            Wc = Wc + 1;
            Wi = [Wi; m];
        end
    else
        if test_set(m,end) ~= I(m)
            Wc = Wc + 1;
            Wi = [Wi; m];
        end
    end
end

% 分类错误率
if k == 0
    disp(['训练集分类错误率为：' num2str(Wc/N*100) '%'])
else
    disp(['测试集test' num2str(k) '分类错误率为：' num2str(Wc/N*100) '%'])
end
% 数据可视化
figure('Name', '训练集数据')
plot(table_female.Height, table_female.Weight, 'ro'), hold on
plot(table_male.Height, table_male.Weight, 'b*')
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male'}, 'Location', 'best')
title('训练集数据')

if k ~= 0
    figure('Name', '测试集数据')
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Gender == ''F''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Gender == ''F''), ''ro'')'])
    hold on
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Gender == ''M''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Gender == ''M''), ''b*'')'])
    xlabel('Height(cm)'), ylabel('Weight(kg)')
    legend({'Female', 'Male'}, 'Location', 'best')
    title('测试集数据')
end

figure('Name', '分类结果')
if k == 0
    plot(table_train.Height(table_train.Classified == 'F'), table_train.Weight(table_train.Classified == 'F'), 'ro'), hold on
    plot(table_train.Height(table_train.Classified == 'M'), table_train.Weight(table_train.Classified == 'M'), 'b*'), hold on
    plot(table_train.Height(Wi), table_train.Weight(Wi), 'ks', 'Markersize', 10), hold on
else
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Classified == ''F''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Classified == ''F''), ''ro'')'])
    hold on
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Classified == ''M''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Classified == ''M''), ''b*'')'])
    hold on
    eval(['plot(table_test' num2str(k) '.Height(Wi), table_test' num2str(k) '.Weight(Wi), ''ks'', ''Markersize'', 10)'])
    hold on
end
fimplicit(fg == 0, 'g -');  %根据方程画分界线
legend({'Female', 'Male', 'Wrong Classification', 'Decision Boundry'}, 'Location', 'best')
xlabel('Height(cm)'), ylabel('Weight(kg)')
title(['测试集:' num2str(k) ' 先验概率(女):' num2str(P_pr(1)) ' 特征:' feature ' 决策准则:' Method '  分类错误率:' num2str(Wc/N*100) '%'])


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
