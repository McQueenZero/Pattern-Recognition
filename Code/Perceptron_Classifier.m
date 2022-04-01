%%-------------------------------------------------------------------------
% ���ߣ�       ������         ѧ�ţ�2018302068     
% ���ڣ�       2021��5��
% ˵����       ���ڵ���֪���ķ�����
% ����汾��   MATLAB R2018a
%%-------------------------------------------------------------------------
%% ����֪���̶����������б���
clc, clear, close all

while 1
    z_C = input('ѡ�����ݼ���1����ҵ�⣬2�������ɣ���') ;
    switch z_C
        case 1
            % ��ҵ��������
            z1 = [0 0; 0 1];
            z2 = [1 0; 1 1];
            N = 2;
            break
        case 2
            % ����������
            N = 5;  %����������
            mu1 = [-1.05 -1.05];
            sigma1 = [1 0.5; 0.5 2];
            R = chol(sigma1);
            z1 = repmat(mu1,N,1) + randn(N,2)*R;  %��һ��
            mu2 = [1.05 1.05];
            sigma2 = [1 0.5; 0.5 1];
            R = chol(sigma2);
            z2 = repmat(mu2,N,1) + randn(N,2)*R;  %�ڶ���
            break
        otherwise
            disp('�Ƿ�������������')
    end
end

for k = 1:N
    % �������� 
    X(:,k) = z1(k,:)';
    X(:,k+N) = z2(k,:)';
    % �������Augmented matrix��
    X_aug(:,k) = [X(:,k); 1];
    X_aug(:,k+N) = [X(:,k+N); 1];
end
      
% ��ʼ����Ȩ�����Ͳ���
W(:,1) = [1 1 1]';
rho = 1; 
m = 1;
k = 1;

plot(X(1,1:N),X(2,1:N),'ro'), hold on
plot(X(1,N+1:N+N),X(2,N+1:N+N),'b*'), hold on
syms x1 x2
name_legend = {'��1' '��2'};

AutoSTOP = 0;
while 1    
    % �������������������Ȩֵ
    while m <= 2*N
        if m <= N && W(:,end)'*X_aug(:,m) <= 0
            W(:,end+1) = W(:,end) + rho * X_aug(:,m);
        end
        if m > N && W(:,end)'*X_aug(:,m) >= 0
            W(:,end+1) = W(:,end) - rho * X_aug(:,m);
        end
        m = m + 1;
    end
    if AutoSTOP == W(:,end)
        break
    end
    AutoSTOP = W(:,end);  %���ֵ���Ȩֵ������һ�ֵ���Ȩֵֹͣ
    if m > 2*N
        m = 1;
    end
    fg = [x1 x2 1] * AutoSTOP;  %�����淽��
    fimplicit(fg == 0)
    name_DB = ['��' num2str(k) '�־�����'];
    name_legend = [name_legend name_DB];
    legend(name_legend, 'Location', 'best')
    hold on
    ManualSTOP = input('��0ֹͣ���س�������');
    if ManualSTOP == 0
        break
    end
    k = k + 1;
end
