%%-------------------------------------------------------------------------
% ���ߣ�       ������         ѧ�ţ�2018302068     
% ���ڣ�       2021��5��
% ˵����       ��Ҷ˹����������⣺����ߺ�/���������ݽ����Ա�����ʵ��
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% ���ļ������ݴ��������������
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
while 1
    P_pr(1,1) = input('��һ��(Ů)�������P_pr(1)=');    %�ֶ������������
    P_pr(1,2) = 1 - P_pr(1,1);
    if 0 <= P_pr(1,1) && P_pr(1,1) <= 1
        break;
    else
        disp('�Ƿ�������ʣ�����������')
    end
end

if k == 0
    test_set = [train_set(:,:,1); train_set(:,:,2)];    %Ӧ�õ�ѵ������
else
    eval(['test_set = test' num2str(k) '_set;']);   %���ݵ�ǰѡ�еĲ��Լ�
end

while 1
    feature = input('���õ�����Ϊ��H/W/B����', 's');    %�ֶ�������õ�����
    %Height/Weight/Both
    switch feature
        case 'H'
            if k ~= 0
                train_set(:,2,:) = [];
                test_set(:,2) = [];
                break;
            else
                disp('�Ƿ�������������')
            end
        case 'W'
            if k ~= 0
                train_set(:,1,:) = [];
                test_set(:,1) = [];
                break;
            else
                disp('�Ƿ�������������')
            end
        case 'B'
            % ����ɾ������
            while 1
                relevant = input('�����Ƿ���أ�Y/N����', 's');   %�ֶ����������Ƿ����
                if relevant == 'N' || relevant == 'n'
                    flag = 0;
                    break;
                elseif relevant == 'Y' || relevant == 'y'
                    flag = 1;
                    break;
                else
                    disp('�Ƿ�������������')
                end
            end           
            break;
        otherwise
            disp('�Ƿ�������������')
    end
end

while 1
    Method = input('����׼����С������/��С���գ�(E/R)��', 's');    %�ֶ�ѡ�񷽷�
    %Error/Risk
    switch Method 
        case 'E'
            break;
        case 'R'
            break;
        otherwise
            disp('�Ƿ�������������')
    end
end

C = size(train_set, 3);     %�����

% ��С������Bayes������
if Method == 'E'
    [P_po, fg_array] = Gauss_BayesClassifier(train_set, test_set, P_pr, flag);  %���غ�����ʺ;��߷�����
    % P_poÿ�д���һ�ࣨ1��ʾŮf��2��ʾ��m��,ÿ�д���һ������
    % ÿ�������ֵ��������������1˵��Ů�Ժ�����ʴ�2˵�����Ժ�����ʴ�
    [~, I] = max(P_po, [], C); 
end

% ��С����Bayes������
% ���߱�Ϊ��
% �������  Ů  ��
% �ж�ΪŮ 0   1.2
% �ж�Ϊ�� 0.8   0
if Method == 'R'
    [P_po, fg_array] = Gauss_BayesClassifier(train_set, test_set, P_pr, flag);  %���غ�����ʺ;��߷�����
    LBD = [0 1.2; 0.8 0]';
    P_R = P_po * LBD;
    fg_array = log(exp(fg_array) * LBD);
    [~, I] = min(P_R, [], C);

end

fg = fg_array(2) - fg_array(1);     %�����淽��
    
% д����Լ���񷽱����
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

% ���������
N = size(test_set, 1);
Wc = 0;     %����������
Wi = [];    %�����������
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

% ���������
if k == 0
    disp(['ѵ�������������Ϊ��' num2str(Wc/N*100) '%'])
else
    disp(['���Լ�test' num2str(k) '���������Ϊ��' num2str(Wc/N*100) '%'])
end
% ���ݿ��ӻ�
figure('Name', 'ѵ��������')
plot(table_female.Height, table_female.Weight, 'ro'), hold on
plot(table_male.Height, table_male.Weight, 'b*')
xlabel('Height(cm)'), ylabel('Weight(kg)')
legend({'Female', 'Male'}, 'Location', 'best')
title('ѵ��������')

if k ~= 0
    figure('Name', '���Լ�����')
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Gender == ''F''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Gender == ''F''), ''ro'')'])
    hold on
    eval(['plot(table_test' num2str(k) '.Height(table_test' num2str(k) '.Gender == ''M''), table_test' num2str(k) '.Weight(table_test' num2str(k) '.Gender == ''M''), ''b*'')'])
    xlabel('Height(cm)'), ylabel('Weight(kg)')
    legend({'Female', 'Male'}, 'Location', 'best')
    title('���Լ�����')
end

figure('Name', '������')
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
fimplicit(fg == 0, 'g -');  %���ݷ��̻��ֽ���
legend({'Female', 'Male', 'Wrong Classification', 'Decision Boundry'}, 'Location', 'best')
xlabel('Height(cm)'), ylabel('Weight(kg)')
title(['���Լ�:' num2str(k) ' �������(Ů):' num2str(P_pr(1)) ' ����:' feature ' ����׼��:' Method '  ���������:' num2str(Wc/N*100) '%'])


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
