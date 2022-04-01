%%-------------------------------------------------------------------------
% 作者：       赵敏琨
% 日期：       2021年6月
% 说明：       对分法聚类作业题
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 初始化
clc
clear
close all
while 1
    X_C = input('选择数据集（1：例题，2：作业题，3：自生成）：') ;
    switch X_C
        case 1
            X = [0 6; 0 5; 2 5; 2 3; 4 4; 4 3; 5 1; 6 2; 6 1; 7 0;
                -4 3; -2 2; -3 2; -3 0; -5 2; 1 1; 0 -1; 0 -2; -1 -1;
                -1 -3; -3 -5];
            break
        case 2
            X = [0 0; 2 2; 1 1; 5 3; 6 3; 5 4; 6 4; 7 5];
            break
        case 3
            % 生成样本集
            N = 50;  %类样本个数
            mu1 = [-1 -1];
            sigma1 = [1 0.5; 0.5 2];
            R = chol(sigma1);
            z1 = repmat(mu1,N,1) + randn(N,2)*R;  %第一类
            mu2 = [1 1];
            sigma2 = [1 0.5; 0.5 1];
            R = chol(sigma2);
            z2 = repmat(mu2,N,1) + randn(N,2)*R;  %第二类
            X = [z1; z2];
            break
        otherwise
            disp('非法，请重新输入')
    end
end

X1.d = X; X2.d = [];  %data
E_threshold = 0;
E_list = [];

%% 对分法
while 1
    X1.m = mean(X1.d); X2.m = mean(X2.d);  %mean
    X1.N = size(X1.d, 1); X2.N = size(X2.d, 1);  %Number of samples
    E = zeros(size(X1.d, 1), 1);
    for ii = 1:size(X1.d, 1)  %每次移一个样本
        temp1.d = [];  temp2.d = [];         
        for jj = 1:size(X1.d, 1)
            if jj ~= ii
                temp1.d = [temp1.d; X1.d(jj, :)];
            end
            temp2.d = [X2.d; X1.d(ii, :)];
        end
        temp1.m = mean(temp1.d); temp2.m = mean(temp2.d);
        temp1.N = size(temp1.d, 1);  temp2.N = size(temp2.d, 1);
        E(ii) = temp1.N*temp2.N/(temp1.N+temp2.N)*(temp1.m-temp2.m)*(temp1.m-temp2.m)';
    end

    [Emax, I] = max(E)
    % 记录每轮最终移动的Emax
    E_list = [E_list Emax];
    
    if Emax < E_threshold
        break
    end
    E_threshold = Emax;
    
    % 更新两类
    X2.d = [X2.d; X1.d(I, :)];  
    X1.d(I, :) = [];

%     close all
%     figure
%     plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
%     plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
%     for k = 1:size(X, 1)
%        text(X(k,1), X(k,2), num2str(k)); 
%     end
end
figure
plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
% for k = 1:size(X, 1)
%    text(X(k,1), X(k,2), num2str(k)); 
% end
legend({'类1', '类2'}, 'Location', 'best')
disp('E_list:')
disp(E_list)

% 凸包问题：画聚类包络线
dt1=delaunayTriangulation(X1.d(:,1), X1.d(:,2));
k1 = convexHull(dt1);
plot(X1.d(k1,1),X1.d(k1,2),'r');
dt2=delaunayTriangulation(X2.d(:,1), X2.d(:,2));
k2 = convexHull(dt2);
plot(X2.d(k2,1),X2.d(k2,2),'b');
