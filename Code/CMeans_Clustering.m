%%-------------------------------------------------------------------------
% ���ߣ�       ������
% ���ڣ�       2021��6��
% ˵����       C��ֵ��������ϰ
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% ��ʼ��
clc
clear
close all
while 1
    X_C = input('ѡ�����ݼ���1�����⣬2�������ɣ���') ;
    switch X_C
        case 1
            X = [0 0; 1 0; 0 1; 1 1; 2 1; 1 2; 2 2; 3 2; 6 6;
                7 6; 8 6; 6 7; 7 7; 8 7; 9 7; 7 8; 8 8; 9 8; 
                8 9; 9 9];
            break
        case 2
            % ����������
            N = 50;  %����������
            mu1 = [-1 -1];
            sigma1 = [1 0.5; 0.5 2];
            R = chol(sigma1);
            z1 = repmat(mu1,N,1) + randn(N,2)*R;  %��һ��
            mu2 = [1 1];
            sigma2 = [1 0.5; 0.5 1];
            R = chol(sigma2);
            z2 = repmat(mu2,N,1) + randn(N,2)*R;  %�ڶ���
            X = [z1; z2];
            break
        otherwise
            disp('�Ƿ�������������')
    end
end

X1.c = X(1, :); X2.c = X(2, :);   %center
C1 = Inf;  C2 = Inf;  %center record

%% C��ֵ��
while 1
    X1.d = []; X2.d = [];  %data
    for ii = 1:size(X, 1)
        if norm(X(ii,:)-X1.c) < norm(X(ii,:)-X2.c)
            X1.d = [X1.d; X(ii, :)];
        else
            X2.d = [X2.d; X(ii, :)];
        end
    end
    X1.c = mean(X1.d);  X2.c = mean(X2.d);  %���¾�������
    disp(['��1����:[' num2str(X1.c(1)) ' ' num2str(X1.c(2)) ']' ...
        ' ��2����:[' num2str(X2.c(1)) ' ' num2str(X2.c(2)) ']'])
    if ~norm(C1-X1.c) && ~norm(C2-X2.c)  %1�ֵ������Ĳ���
        break
    end
    C1 = X1.c; C2 = X2.c;  %��¼��������
    
%     close all
%     figure
%     plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
%     plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
end
figure
plot(X1.d(:,1), X1.d(:,2), 'ro'), hold on
plot(X2.d(:,1), X2.d(:,2), 'b*'), hold on
% for k = 1:size(X, 1)
%    text(X(k,1), X(k,2), num2str(k)); 
% end
% legend({'��1', '��2'}, 'Location', 'best')

% ͹�����⣺�����������
dt1=delaunayTriangulation(X1.d(:,1), X1.d(:,2));
k1 = convexHull(dt1);
plot(X1.d(k1,1),X1.d(k1,2),'r');
dt2=delaunayTriangulation(X2.d(:,1), X2.d(:,2));
k2 = convexHull(dt2);
plot(X2.d(k2,1),X2.d(k2,2),'b');
