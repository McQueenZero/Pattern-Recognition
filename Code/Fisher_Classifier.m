%%-------------------------------------------------------------------------
% ���ߣ�       ������         ѧ�ţ�2018302068     
% ���ڣ�       2021��6��
% ˵����       ����Fisher׼��ķ�����
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
    if k ~= 1 && k ~= 2 && k~=0
        disp('�Ƿ���ţ�����������')
    else
        break;
    end
end
if k == 0
    test_set = [train_set(:,:,1) ones(size(train_set(:,:,1), 1), 1); ...
        train_set(:,:,2) 2*ones(size(train_set(:,:,1), 1), 1)];    %Ӧ�õ�ѵ������
else
    eval(['test_set = test' num2str(k) '_set;']);   %���ݵ�ǰѡ�еĲ��Լ�
end

C = size(train_set, 3);     %�����

%% Fisher����׼��
X.f = test_set(test_set(:,3)==1, 1:2);
X.m = test_set(test_set(:,3)==2, 1:2);
X.d = test_set(:, 1:2);
X.f_bar = mean(X.f, 1);  %���Լ�Ů�Եľ�ֵ
X.m_bar = mean(X.m, 1);  %���Լ����Եľ�ֵ
X.f_S = 0;  %���Լ�Ů�Եķ������
X.m_S = 0;  %���Լ����Եķ������
for ii = 1:size(X.f, 1)
    X.f_S = X.f_S + (X.f(ii, :) - X.f_bar)' * (X.f(ii, :) - X.f_bar);
end
X.f_S = 1 / size(X.f, 1) * X.f_S;  %��ѧ����
for ii = 1:size(X.m, 1)
    X.m_S = X.m_S + (X.m(ii, :) - X.m_bar)' * (X.m(ii, :) - X.m_bar);
end  
X.m_S = 1 / size(X.m, 1) * X.m_S;  %��ѧ����
% ע��һ����������ƫЭ�������=���������������(��)����ѧ����
% X.f_S = cov(X.f);  %���Լ�Ů�Ե���ƫЭ�������
% X.m_S = cov(X.m);  %���Լ����Ե���ƫЭ�������
S_b = (X.f_bar - X.m_bar)' * (X.f_bar - X.m_bar);  %���ɢ������
S_w = X.f_S + X.m_S;  %����ɢ������
W = S_w^(-1) * (X.f_bar - X.m_bar)';  %��άӳ��
Ax = W' * eye(2); %һά������
L.f = W' * X.f';  %һά���꣬�����һάԭ��ľ���
L.m = W' * X.m';  
L.d = W' * X.d';

% һά�ľ�ֵ��������ֵ
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

for kk = 1:size(L.d, 2)  %��һά����
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
f_x = Ax(2)/Ax(1) * x;  %һά������

Y.f = zeros(2, size(L.f, 2));
% ����һά����Ķ�ά��ʾ
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

% ����һά��ֵ�Ķ�ά��ʾ
eqn.thr1 = x^2 + f_x^2 == L.thr1 .^ 2;
eqn.thr2 = x^2 + f_x^2 == L.thr2 .^ 2;
eqn.thr3 = x^2 + f_x^2 == L.thr3 .^ 2;
Y.thr1 = solve(eqn.thr1, x);
Y.thr2 = solve(eqn.thr2, x);
Y.thr3 = solve(eqn.thr3, x);
Y.thr1(1) = [];
Y.thr2(1) = [];
Y.thr3(1) = [];
% �õ�������
f_B1 = -Ax(1)/Ax(2) * (x - Y.thr1) + subs(f_x, Y.thr1);
f_B2 = -Ax(1)/Ax(2) * (x - Y.thr2) + subs(f_x, Y.thr2);
f_B3 = -Ax(1)/Ax(2) * (x - Y.thr3) + subs(f_x, Y.thr3);
% Y.f��Y.f_y������ĺ��ݷ���
Y.f_y = eval(subs(f_x, Y.f));
Y.m_y = eval(subs(f_x, Y.m));

% ���������
N = size(test_set, 1);
Wc1 = 0;     %����������
Wi1 = [];    %�����������
Wc2 = 0;     %����������
Wi2 = [];    %�����������
Wc3 = 0;     %����������
Wi3 = [];    %�����������
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

fplot(f_x), hold on  %һά������
plot(Y.f, Y.f_y, 'ro')
plot(Y.m, Y.m_y, 'b*')
fplot(f_B1), fplot(f_B2), fplot(f_B3)
axis([min(Y.f) max(Y.m) ...
    (min(Y.f_y)+max(Y.m_y))/2-(max(Y.m)-min(Y.f))/2 ...
    (min(Y.f_y)+max(Y.m_y))/2+(max(Y.m)-min(Y.f))/2])
legend({'һά������', 'Female', 'Male', '������1', '������2', '������3'}, 'Location', 'best')
title('��ά����')

figure
subplot(2,2,1)
plot(X.f(:, 1), X.f(:, 2), 'ro'), hold on
plot(X.m(:, 1), X.m(:, 2), 'b*')
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male'}, 'Location', 'best')
title('��������')

subplot(2,2,2)
plot(X.c.f1(:, 1), X.c.f1(:, 2), 'ro'), hold on
plot(X.c.m1(:, 1), X.c.m1(:, 2), 'b*')
plot(X.d(Wi1, 1), X.d(Wi1, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['��ֵ1��' num2str(L.thr1) '�Ľ��' '�������ʣ�' num2str(Wc1/N*100) '%'])
subplot(2,2,3)
plot(X.c.f2(:, 1), X.c.f2(:, 2), 'ro'), hold on
plot(X.c.m2(:, 1), X.c.m2(:, 2), 'b*')
plot(X.d(Wi2, 1), X.d(Wi2, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['��ֵ2��' num2str(L.thr2) '�Ľ��' '�������ʣ�' num2str(Wc2/N*100) '%'])
subplot(2,2,4)
plot(X.c.f3(:, 1), X.c.f3(:, 2), 'ro'), hold on
plot(X.c.m3(:, 1), X.c.m3(:, 2), 'b*')
plot(X.d(Wi3, 1), X.d(Wi3, 2), 'ks', 'Markersize', 10)
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male', 'Error'}, 'Location', 'best')
title(['��ֵ3��' num2str(L.thr3) '�Ľ��' '�������ʣ�' num2str(Wc3/N*100) '%'])

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
