%%-------------------------------------------------------------------------
% �޸��ߣ�     ������  
% ���ڣ�       2021��6��
% ˵����       K-L�任����ʶ��
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% ���ݴ���
clc
clear
close all
num_train = 360;
for ii = 1:num_train
    data_path = (['.\eigfaces\train_set\' num2str(ii) '.bmp']); 
    img = imread(data_path);
    %img = imnoise(img, 'salt & pepper');
    train_set(:, ii) = double(img(:));
end

train_mean = mean(train_set')';  %ѵ����
train_mean = repmat(train_mean, 1 ,num_train);
X_NNxM = train_set - train_mean;
R_mxm = X_NNxM' * X_NNxM;  %��������
[v, d] = eig(R_mxm);  %����ֵ�ֽ����������������ֵ
% ���������ԭ��������ֵ��ͬ����������Ϊ�¹�ϵ
v_Sig = X_NNxM * v;  %Sigma�������������
[lambda, I] = sort(diag(d), 'descend');  %����ֵ����
v_Sig_sort = v_Sig(:, I);  %��������Ҳ������ֵ��С��������
den = zeros(num_train, 1);
for jj = 1:num_train
    den(jj, 1) = lambda(jj, 1)^(-1/2);
    v_Sig_sort_norm(:,jj) = v_Sig_sort(:,jj) * den(jj, 1);  %����������һ��
end

%% ѡ������������Ϊthr������ֵ
while 1
    thr = input('����������ֵ��ֵ��');
    if thr <= 0 || thr > 1
        disp('�Ƿ�������������')
    else
        break
    end
end

% eigface_n_array = [];
% correct_R_array = [];
% for thr = 0.01:0.01:0.99

eig_all = sum(lambda);  %����ֵ�ܺ�
eig_thr = eig_all * thr;  %����ֵ����ֵ
W = 0;
for ii = 1:num_train
    W = W + lambda(ii, 1);
    if W >= eig_thr
        eigface_n = ii;
        break
    end
end

disp(['������������' num2str(eigface_n)])
% ��ϵ������ѡȡǰeigface_n������������
C_ratio = v_Sig_sort_norm(:, 1:eigface_n)' * X_NNxM;  %C_ratio��ʾ��i���������ڵ�j�ű�������ϵ��

% ��ʾ������
% for ii = 1:eigface_n

for ii = 3:18  %���ֱ�����
    eigface = reshape(v_Sig_sort_norm(:, ii), 112, 92);
    f_lap = fspecial('laplacian', 0);
    imfilter(eigface, f_lap);
    uint8(eigface);
    % subplot(5, 5, ii)
    subplot(4, 4, ii-2)
    histeq(eigface)
end

%% ����
correct_C = 0;  %��ȷ����

num_test = 40;
for ii = 1:num_test
    data_path = (['.\eigfaces\test_set\' num2str(ii) '.bmp']); 
    img = imread(data_path);
    test_set(:, 1) = double(img(:));
    test_mean = test_set - train_mean(:, 1);
    C = v_Sig_sort_norm(:, 1:eigface_n)' * test_mean;
    for jj = 1:num_train
        D(jj) = sqrt(sum((C - C_ratio(:, jj)).^2));
    end
    [D_sort, D_I] = sort(D, 'ascend');
    % ���ѵ�����ĸ������ı�ź͵�ǰ��������ŷ�Ͼ�����С
    person_n = ceil(D_I(1) / 9); %ѵ����һ����9������
    if person_n == ii
        Judge = 'True';
        correct_C = correct_C + 1;
    else
        Judge = 'False';
    end
    %if strcmp(Judge, 'False')
    disp(['���Լ���ţ�Person' num2str(ii) '��Ԥ���ţ�Person' num2str(person_n) '���ж�: ' Judge])
    %end
end
correct_R = (correct_C / num_test) * 100;  %��ȷ����
disp(['��ȷ������' num2str(correct_C) '����ȷ���ʣ�' num2str(correct_R) '%'])

% eigface_n_array = [eigface_n_array eigface_n];
% correct_R_array = [correct_R_array correct_R];
% end
% figure
% subplot(1,2,1)
% stairs(0.01:0.01:0.99, eigface_n_array)
% xlabel('��ֵ'), ylabel('����������')
% title('��ֵ�ͱ����������Ĺ�ϵ')
% subplot(1,2,2)
% plot(0.01:0.01:0.99, correct_R_array)
% xlabel('��ֵ'), ylabel('��ȷ��(%)')
% title('��ֵ��ʶ����ȷ�ʵĹ�ϵ')
% figure
% plot(eigface_n_array, correct_R_array)
% xlabel('����������'), ylabel('��ȷ��(%)')
% title('������������ʶ����ȷ�ʵĹ�ϵ')
