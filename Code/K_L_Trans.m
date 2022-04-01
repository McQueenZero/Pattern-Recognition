%%-------------------------------------------------------------------------
% 修改者：     赵敏琨  
% 日期：       2021年6月
% 说明：       K-L变换人脸识别
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 数据处理
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

train_mean = mean(train_set')';  %训练集
train_mean = repmat(train_mean, 1 ,num_train);
X_NNxM = train_set - train_mean;
R_mxm = X_NNxM' * X_NNxM;  %产生矩阵
[v, d] = eig(R_mxm);  %奇异值分解得特征向量和特征值
% 产生矩阵和原矩阵特征值相同，特征向量为下关系
v_Sig = X_NNxM * v;  %Sigma矩阵的特征向量
[lambda, I] = sort(diag(d), 'descend');  %特征值排序
v_Sig_sort = v_Sig(:, I);  %特征向量也按特征值大小降序排列
den = zeros(num_train, 1);
for jj = 1:num_train
    den(jj, 1) = lambda(jj, 1)^(-1/2);
    v_Sig_sort_norm(:,jj) = v_Sig_sort(:,jj) * den(jj, 1);  %特征向量归一化
end

%% 选择特征贡献率为thr的特征值
while 1
    thr = input('请输入特征值阈值：');
    if thr <= 0 || thr > 1
        disp('非法，请重新输入')
    else
        break
    end
end

% eigface_n_array = [];
% correct_R_array = [];
% for thr = 0.01:0.01:0.99

eig_all = sum(lambda);  %特征值总和
eig_thr = eig_all * thr;  %特征值的阈值
W = 0;
for ii = 1:num_train
    W = W + lambda(ii, 1);
    if W >= eig_thr
        eigface_n = ii;
        break
    end
end

disp(['本征脸个数：' num2str(eigface_n)])
% 求系数矩阵，选取前eigface_n张脸作本征脸
C_ratio = v_Sig_sort_norm(:, 1:eigface_n)' * X_NNxM;  %C_ratio表示第i张人脸对于第j张本征脸的系数

% 显示本征脸
% for ii = 1:eigface_n

for ii = 3:18  %部分本征脸
    eigface = reshape(v_Sig_sort_norm(:, ii), 112, 92);
    f_lap = fspecial('laplacian', 0);
    imfilter(eigface, f_lap);
    uint8(eigface);
    % subplot(5, 5, ii)
    subplot(4, 4, ii-2)
    histeq(eigface)
end

%% 测试
correct_C = 0;  %正确计数

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
    % 输出训练集哪个人脸的编号和当前测试人脸欧氏距离最小
    person_n = ceil(D_I(1) / 9); %训练集一个人9张人脸
    if person_n == ii
        Judge = 'True';
        correct_C = correct_C + 1;
    else
        Judge = 'False';
    end
    %if strcmp(Judge, 'False')
    disp(['测试集编号：Person' num2str(ii) '，预测编号：Person' num2str(person_n) '，判断: ' Judge])
    %end
end
correct_R = (correct_C / num_test) * 100;  %正确比率
disp(['正确计数：' num2str(correct_C) '，正确比率：' num2str(correct_R) '%'])

% eigface_n_array = [eigface_n_array eigface_n];
% correct_R_array = [correct_R_array correct_R];
% end
% figure
% subplot(1,2,1)
% stairs(0.01:0.01:0.99, eigface_n_array)
% xlabel('阈值'), ylabel('本征脸个数')
% title('阈值和本征脸个数的关系')
% subplot(1,2,2)
% plot(0.01:0.01:0.99, correct_R_array)
% xlabel('阈值'), ylabel('正确率(%)')
% title('阈值和识别正确率的关系')
% figure
% plot(eigface_n_array, correct_R_array)
% xlabel('本征脸个数'), ylabel('正确率(%)')
% title('本征脸个数和识别正确率的关系')
