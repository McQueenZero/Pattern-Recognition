%%-------------------------------------------------------------------------
% 作者：       赵敏琨         学号：2018302068     
% 日期：       2021年6月
% 说明：       基于支持向量机(SVM)的分类器
% 软件版本：   MATLAB R2018a
% 注意：
%   运行测试集0时间较长，运行测试集2时间很长，是由于Matlab的
%   符号代换导致的，为了节省时间，我将第一次运行生成的优化问题保存，
%   后续只需先后运行第一节(数据处理)和第三节(优化问题求解)即可   
%   （即便如此，由于测试集2数据多，耗时仍然较长，打断点发现
%   主要耗时在fmincon的初始化上，2分钟左右）
%   若老师/助教运行此代码，为省时请选测试集1或3
%   选择测试集3需要每次都保存优化问题，因为每次生成的测试数据是随机的
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
    if k ~= 1 && k ~= 2 && k~=0  && k~=3
        disp('非法编号，请重新输入')
    else
        break;
    end
end
if k == 0
    test_set = [train_set(:,:,1) ones(size(train_set(:,:,1), 1), 1); ...
        train_set(:,:,2) 2*ones(size(train_set(:,:,1), 1), 1)];    %应用到训练集上
elseif k == 1 || k == 2
    eval(['test_set = test' num2str(k) '_set;']);   %传递当前选中的测试集
else
    % 生成样本集
    N = 10;  %类样本个数
    mu1 = [-1.05 -1.05];
    sigma1 = [1 0.5; 0.5 2];
    R = chol(sigma1);
    z1 = repmat(mu1,N,1) + randn(N,2)*R;  %第一类
    mu2 = [1.05 1.05];
    sigma2 = [1 0.5; 0.5 1];
    R = chol(sigma2);
    z2 = repmat(mu2,N,1) + randn(N,2)*R;  %第二类
    test_set = [z1 ones(N,1); z2 2*ones(N,1)];
end

% 改成正例负例点
test_set(test_set(:,3)==2, 3) = -1;

N = size(test_set, 1);
a = sym('a', [N, 1]);

%% 支持向量机
% 对偶问题
a_sum = 0; ayx_sum = 0; ay_sum = 0;
for ii = 1:N
    a_sum = a_sum + a(ii);
    ay_sum = ay_sum + a(ii)*test_set(ii,3);
    for jj = 1:N
        ayx_sum = ayx_sum + ...
            a(ii)*a(jj)*test_set(ii,3)*test_set(jj,3)* ...
            (test_set(ii,1:2)*test_set(jj,1:2)');
    end
end
g_obj = 0.5 * ayx_sum - a_sum;
g_st.e = ay_sum;
% 硬间隔
g_st.ieh = -a;
% 软间隔
C = 20;
g_st.ies = a - C;

g_obj
argsym_old = [];
argsym_new = [];
for ii = 1:N
    argsym_old = {[argsym_old a(ii)]};
    argsym_new = {[argsym_new str2sym(['A(' int2str(ii) ')'])]};
end
g_obj = subs(g_obj, argsym_old{:}, argsym_new{:});
g_st.e = subs(g_st.e, argsym_old{:}, argsym_new{:});
g_st.ieh = subs(g_st.ieh, argsym_old{:}, argsym_new{:});
g_st.ies = subs(g_st.ies, argsym_old{:}, argsym_new{:});

g_obj
syms A

SAVEFLAG = input('输入''s''保存优化问题，输入其他跳过：', 's');
if SAVEFLAG == 's'
    switch k
        case 0
            matlabFunction(g_obj, 'File', 'SVM_opt_fcn_test0', 'Vars', 'A');
            matlabFunction(g_st.ies, g_st.e, 'File', 'SVM_opt_cons_test0', 'Vars', 'A', 'Outputs', {'g','h'});
        case 1
            matlabFunction(g_obj, 'File', 'SVM_opt_fcn_test1', 'Vars', 'A');
            matlabFunction(g_st.ies, g_st.e, 'File', 'SVM_opt_cons_test1', 'Vars', 'A', 'Outputs', {'g','h'});
        case 2
            SVM_opt_fcn_test2 = matlabFunction(g_obj, 'Vars', 'A');
            SVM_opt_cons_test2 = matlabFunction(g_st.ies, g_st.e, 'Vars', 'A', 'Outputs', {'g','h'});
            save SVM_opt_test2 g_obj g_st SVM_opt_fcn_test2 SVM_opt_cons_test2
        case 3
            matlabFunction(g_obj, 'File', 'SVM_opt_fcn_test3', 'Vars', 'A');
            matlabFunction(g_st.ies, g_st.e, 'File', 'SVM_opt_cons_test3', 'Vars', 'A', 'Outputs', {'g','h'});
    end 
end

%% 优化问题求解
a0 = zeros(size(a));
lb = zeros(size(a))';
hb = [];
% hb = 50*ones(size(a))';
if k == 2
    load('SVM_opt_test2.mat')
    SVM_opt_fcn = SVM_opt_fcn_test2;
    SVM_opt_cons = SVM_opt_cons_test2;
else
    eval(['SVM_opt_fcn = @SVM_opt_fcn_test' int2str(k) ';'])
    eval(['SVM_opt_cons = @SVM_opt_cons_test' int2str(k) ';'])
end
opts = optimoptions('fmincon','PlotFcn','optimplotfval','OptimalityTolerance',1e-10,'MaxFunctionEvaluations',5000);
a_opt = fmincon(SVM_opt_fcn, a0, [], [], [], [], lb, hb, SVM_opt_cons, opts);
% close all
w = 0; yax_sum = 0;
% jj = randi(N,1,1); %jj取满足KKT条件的a_j下标
[~, jj] = max(a_opt)  %jj代表支持向量的索引
% [~, jj] = min(a_opt)  %jj代表支持向量的索引
% [~, jj] = sort(a_opt)  %jj代表支持向量的索引
% jj = jj(round(N/2));
syms x1 x2
x = [x1; x2];
% for jj = N:-5:1
for ii = 1:N
    w = w + a_opt(ii)*test_set(ii,3).*test_set(ii,1:2);
    yax_sum = yax_sum + test_set(ii,3)*a_opt(ii)*test_set(ii,1:2)*test_set(jj,1:2)';
end   
b = test_set(jj,3) - yax_sum;
f_x = w * x + b
y = solve(f_x==0, x2);
for ii = 1:N
    test_set(ii,4) = sign(subs(f_x, [x1; x2], test_set(ii,1:2)'));
end

% 计算错误率
Wc = 0;     %分类错误计数
Wi = [];    %分类错误索引
for m = 1:N
    if test_set(m,4) ~= test_set(m,3)
        Wc = Wc + 1;
        Wi = [Wi; m];
    end
end

% 可视化
figure
fplot(y), hold on
% end
plot(test_set((test_set(:,3)==1),1), test_set((test_set(:,3)==1),2), 'ro'), hold on
plot(test_set((test_set(:,3)==-1),1), test_set((test_set(:,3)==-1),2), 'b*'), hold on
legend({'Decision Boundry', 'Female', 'Male'}, 'Location', 'best')
% 手动找支持向量索引(报告中的图3-9)
% plot(test_set([3 9 15],1), test_set([3 9 15],2), 'ks', 'Markersize', 10)
% legend({'Decision Boundry', 'Female', 'Male', 'Support Vector'}, 'Location', 'best')
title('测试数据和决策面')
axis([min(test_set(:,1)) max(test_set(:,1)) min(test_set(:,2)) max(test_set(:,2))])
figure
plot(test_set((test_set(:,4)==1),1), test_set((test_set(:,4)==1),2), 'ro'), hold on
plot(test_set((test_set(:,4)==-1),1), test_set((test_set(:,4)==-1),2), 'b*'), hold on
plot(test_set(Wi,1), test_set(Wi,2), 'ks', 'Markersize', 10)
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['分类结果，错误率：' num2str(Wc/N*100) '%'])
axis([min(test_set(:,1)) max(test_set(:,1)) min(test_set(:,2)) max(test_set(:,2))])

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
