%%-------------------------------------------------------------------------
% 作者：       赵敏琨         学号：2018302068     
% 日期：       2021年6月
% 说明：       基于Fisher准则的分类器
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
if k == 0
    test_set = [train_set(:,:,1) ones(size(train_set(:,:,1), 1), 1); ...
        train_set(:,:,2) 2*ones(size(train_set(:,:,1), 1), 1)];    %应用到训练集上
else
    eval(['test_set = test' num2str(k) '_set;']);   %传递当前选中的测试集
end

C = size(train_set, 3);     %类别数

%% Fisher分类准则
X.f = test_set(test_set(:,3)==1, 1:2);
X.m = test_set(test_set(:,3)==2, 1:2);
X.d = test_set(:, 1:2);
X.f_bar = mean(X.f, 1);  %测试集女性的均值
X.m_bar = mean(X.m, 1);  %测试集男性的均值
X.f_S = 0;  %测试集女性的方差矩阵
X.m_S = 0;  %测试集男性的方差矩阵
for ii = 1:size(X.f, 1)
    X.f_S = X.f_S + (X.f(ii, :) - X.f_bar)' * (X.f(ii, :) - X.f_bar);
end
X.f_S = 1 / size(X.f, 1) * X.f_S;  %数学期望
for ii = 1:size(X.m, 1)
    X.m_S = X.m_S + (X.m(ii, :) - X.m_bar)' * (X.m(ii, :) - X.m_bar);
end  
X.m_S = 1 / size(X.m, 1) * X.m_S;  %数学期望
% 注：一类样本的有偏协方差矩阵=该类样本方差矩阵(和)的数学期望
% X.f_S = cov(X.f);  %测试集女性的无偏协方差矩阵
% X.m_S = cov(X.m);  %测试集男性的无偏协方差矩阵
S_b = (X.f_bar - X.m_bar)' * (X.f_bar - X.m_bar);  %类间散布矩阵
S_w = X.f_S + X.m_S;  %类内散布矩阵
W = S_w^(-1) * (X.f_bar - X.m_bar)';  %降维映射
Ax = W' * eye(2); %一维基向量
L.f = W' * X.f';  %一维坐标，即点和一维原点的距离
L.m = W' * X.m';  
L.d = W' * X.d';

% 一维的均值和三种阈值
L.f_bar = mean(L.f);
L.m_bar = mean(L.m);
L.thr1 = (L.f_bar + L.m_bar)/2;
L.thr2 = (size(L.f, 2) * L.f_bar + size(L.m, 2) * L.m_bar) / (size(L.f, 2) + size(L.m, 2));
L.thr3 = L.f_bar + (L.m_bar - L.f_bar) * var(L.f) / (var(L.f) + var(L.m));

L.c.f1 = []; L.c.m1 = [];
L.c.f2 = []; L.c.m2 = [];
L.c.f3 = []; L.c.m3 = [];
L.c.I1 = zeros(size(L.d));
L.c.I2 = zeros(size(L.d));
L.c.I3 = zeros(size(L.d));

for kk = 1:size(L.d, 2)  %在一维分类
    if L.d(kk) > L.thr1
        L.c.f1 = [L.c.f1 L.d(kk)];
        L.c.I1(kk) = 1;
    else
        L.c.m1 = [L.c.m1 L.d(kk)];
        L.c.I1(kk) = 2;
    end
    if L.d(kk) > L.thr2
        L.c.f2 = [L.c.f2 L.d(kk)];
        L.c.I2(kk) = 1;
    else
        L.c.m2 = [L.c.m2 L.d(kk)];
        L.c.I2(kk) = 2;
    end
    if L.d(kk) > L.thr3
        L.c.f3 = [L.c.f3 L.d(kk)];
        L.c.I3(kk) = 1;
    else
        L.c.m3 = [L.c.m3 L.d(kk)];
        L.c.I3(kk) = 2;
    end
end

X.c.f1 = X.d(L.c.I1==1, :);
X.c.m1 = X.d(L.c.I1==2, :);
X.c.f2 = X.d(L.c.I2==1, :);
X.c.m2 = X.d(L.c.I2==2, :);
X.c.f3 = X.d(L.c.I3==1, :);
X.c.m3 = X.d(L.c.I3==2, :);

syms x
f_x = Ax(2)/Ax(1) * x;  %一维坐标轴

Y.f = zeros(2, size(L.f, 2));
% 反解一维坐标的二维表示
for nn = 1:size(L.f, 2)
    eqn.f = x^2 + f_x^2 == L.f(nn) .^ 2;
    Y.f(:, nn) = solve(eqn.f, x);
end
Y.m = zeros(2, size(L.m, 2));
for nn = 1:size(L.m, 2)
    eqn.m = x^2 + f_x^2 == L.m(nn) .^ 2;
    Y.m(:, nn) = solve(eqn.m, x);
end
Y.f(1, :) = [];
Y.m(1, :) = [];

% 反解一维阈值的二维表示
eqn.thr1 = x^2 + f_x^2 == L.thr1 .^ 2;
eqn.thr2 = x^2 + f_x^2 == L.thr2 .^ 2;
eqn.thr3 = x^2 + f_x^2 == L.thr3 .^ 2;
Y.thr1 = solve(eqn.thr1, x);
Y.thr2 = solve(eqn.thr2, x);
Y.thr3 = solve(eqn.thr3, x);
Y.thr1(1) = [];
Y.thr2(1) = [];
Y.thr3(1) = [];
% 得到决策面
f_B1 = -Ax(1)/Ax(2) * (x - Y.thr1) + subs(f_x, Y.thr1);
f_B2 = -Ax(1)/Ax(2) * (x - Y.thr2) + subs(f_x, Y.thr2);
f_B3 = -Ax(1)/Ax(2) * (x - Y.thr3) + subs(f_x, Y.thr3);
% Y.f和Y.f_y是坐标的横纵分量
Y.f_y = eval(subs(f_x, Y.f));
Y.m_y = eval(subs(f_x, Y.m));

% 计算错误率
N = size(test_set, 1);
Wc1 = 0;     %分类错误计数
Wi1 = [];    %分类错误索引
Wc2 = 0;     %分类错误计数
Wi2 = [];    %分类错误索引
Wc3 = 0;     %分类错误计数
Wi3 = [];    %分类错误索引
for m = 1:N
    if test_set(m,end) ~= L.c.I1(m)
        Wc1 = Wc1 + 1;
        Wi1 = [Wi1; m];
    end
    if test_set(m,end) ~= L.c.I2(m)
        Wc2 = Wc2 + 1;
        Wi2 = [Wi2; m];
    end
    if test_set(m,end) ~= L.c.I3(m)
        Wc3 = Wc3 + 1;
        Wi3 = [Wi3; m];
    end
end

fplot(f_x), hold on  %一维坐标轴
plot(Y.f, Y.f_y, 'ro')
plot(Y.m, Y.m_y, 'b*')
fplot(f_B1), fplot(f_B2), fplot(f_B3)
axis([min(Y.f) max(Y.m) ...
    (min(Y.f_y)+max(Y.m_y))/2-(max(Y.m)-min(Y.f))/2 ...
    (min(Y.f_y)+max(Y.m_y))/2+(max(Y.m)-min(Y.f))/2])
legend({'一维坐标轴', 'Female', 'Male', '决策面1', '决策面2', '决策面3'}, 'Location', 'best')
title('降维数据')

figure
subplot(2,2,1)
plot(X.f(:, 1), X.f(:, 2), 'ro'), hold on
plot(X.m(:, 1), X.m(:, 2), 'b*')
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male'}, 'Location', 'best')
title('测试数据')

subplot(2,2,2)
plot(X.c.f1(:, 1), X.c.f1(:, 2), 'ro'), hold on
plot(X.c.m1(:, 1), X.c.m1(:, 2), 'b*')
plot(X.d(Wi1, 1), X.d(Wi1, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['阈值1：' num2str(L.thr1) '的结果' '，错误率：' num2str(Wc1/N*100) '%'])
subplot(2,2,3)
plot(X.c.f2(:, 1), X.c.f2(:, 2), 'ro'), hold on
plot(X.c.m2(:, 1), X.c.m2(:, 2), 'b*')
plot(X.d(Wi2, 1), X.d(Wi2, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['阈值2：' num2str(L.thr2) '的结果' '，错误率：' num2str(Wc2/N*100) '%'])
subplot(2,2,4)
plot(X.c.f3(:, 1), X.c.f3(:, 2), 'ro'), hold on
plot(X.c.m3(:, 1), X.c.m3(:, 2), 'b*')
plot(X.d(Wi3, 1), X.d(Wi3, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['阈值3：' num2str(L.thr3) '的结果' '，错误率：' num2str(Wc3/N*100) '%'])

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
