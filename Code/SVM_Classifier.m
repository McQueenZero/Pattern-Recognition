%%-------------------------------------------------------------------------
% ���ߣ�       ������         ѧ�ţ�2018302068     
% ���ڣ�       2021��6��
% ˵����       ����֧��������(SVM)�ķ�����
% ����汾��   MATLAB R2018a
% ע�⣺
%   ���в��Լ�0ʱ��ϳ������в��Լ�2ʱ��ܳ���������Matlab��
%   ���Ŵ������µģ�Ϊ�˽�ʡʱ�䣬�ҽ���һ���������ɵ��Ż����Ᵽ�棬
%   ����ֻ���Ⱥ����е�һ��(���ݴ���)�͵�����(�Ż��������)����   
%   ��������ˣ����ڲ��Լ�2���ݶ࣬��ʱ��Ȼ�ϳ�����ϵ㷢��
%   ��Ҫ��ʱ��fmincon�ĳ�ʼ���ϣ�2�������ң�
%   ����ʦ/�������д˴��룬Ϊʡʱ��ѡ���Լ�1��3
%   ѡ����Լ�3��Ҫÿ�ζ������Ż����⣬��Ϊÿ�����ɵĲ��������������
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

% ѵ������������ת���������һҳ��Ů�ԣ�����ڶ�ҳ������
train_set(:,:,1) = [table_female.Height table_female.Weight];
train_set(:,:,2) = [table_male.Height table_male.Weight];

% ���Լ���������ת�����������һ�����Ա�1��ʾŮ�ԣ�2��ʾ���ԣ�
test1_set = [table_test1.Height table_test1.Weight];
test2_set = [table_test2.Height table_test2.Weight];
test1_set(table_test1.Gender == 'F', 3) = 1;
test1_set(table_test1.Gender == 'M', 3) = 2;
test2_set(table_test2.Gender == 'F', 3) = 1;
test2_set(table_test2.Gender == 'M', 3) = 2;

while 1
    k = input('ѡ����Լ����=');    %�ֶ�����ѡ����Լ����
    if k ~= 1 && k ~= 2 && k~=0  && k~=3
        disp('�Ƿ���ţ�����������')
    else
        break;
    end
end
if k == 0
    test_set = [train_set(:,:,1) ones(size(train_set(:,:,1), 1), 1); ...
        train_set(:,:,2) 2*ones(size(train_set(:,:,1), 1), 1)];    %Ӧ�õ�ѵ������
elseif k == 1 || k == 2
    eval(['test_set = test' num2str(k) '_set;']);   %���ݵ�ǰѡ�еĲ��Լ�
else
    % ����������
    N = 10;  %����������
    mu1 = [-1.05 -1.05];
    sigma1 = [1 0.5; 0.5 2];
    R = chol(sigma1);
    z1 = repmat(mu1,N,1) + randn(N,2)*R;  %��һ��
    mu2 = [1.05 1.05];
    sigma2 = [1 0.5; 0.5 1];
    R = chol(sigma2);
    z2 = repmat(mu2,N,1) + randn(N,2)*R;  %�ڶ���
    test_set = [z1 ones(N,1); z2 2*ones(N,1)];
end

% �ĳ�����������
test_set(test_set(:,3)==2, 3) = -1;

N = size(test_set, 1);
a = sym('a', [N, 1]);

%% ֧��������
% ��ż����
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
% Ӳ���
g_st.ieh = -a;
% ����
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

SAVEFLAG = input('����''s''�����Ż����⣬��������������', 's');
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

%% �Ż��������
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
% jj = randi(N,1,1); %jjȡ����KKT������a_j�±�
[~, jj] = max(a_opt)  %jj����֧������������
% [~, jj] = min(a_opt)  %jj����֧������������
% [~, jj] = sort(a_opt)  %jj����֧������������
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

% ���������
Wc = 0;     %����������
Wi = [];    %�����������
for m = 1:N
    if test_set(m,4) ~= test_set(m,3)
        Wc = Wc + 1;
        Wi = [Wi; m];
    end
end

% ���ӻ�
figure
fplot(y), hold on
% end
plot(test_set((test_set(:,3)==1),1), test_set((test_set(:,3)==1),2), 'ro'), hold on
plot(test_set((test_set(:,3)==-1),1), test_set((test_set(:,3)==-1),2), 'b*'), hold on
legend({'Decision Boundry', 'Female', 'Male'}, 'Location', 'best')
% �ֶ���֧����������(�����е�ͼ3-9)
% plot(test_set([3 9 15],1), test_set([3 9 15],2), 'ks', 'Markersize', 10)
% legend({'Decision Boundry', 'Female', 'Male', 'Support Vector'}, 'Location', 'best')
title('�������ݺ;�����')
axis([min(test_set(:,1)) max(test_set(:,1)) min(test_set(:,2)) max(test_set(:,2))])
figure
plot(test_set((test_set(:,4)==1),1), test_set((test_set(:,4)==1),2), 'ro'), hold on
plot(test_set((test_set(:,4)==-1),1), test_set((test_set(:,4)==-1),2), 'b*'), hold on
plot(test_set(Wi,1), test_set(Wi,2), 'ks', 'Markersize', 10)
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['�������������ʣ�' num2str(Wc/N*100) '%'])
axis([min(test_set(:,1)) max(test_set(:,1)) min(test_set(:,2)) max(test_set(:,2))])

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
