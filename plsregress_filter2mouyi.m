function [train_err,test_err,H,train_corr,test_corr]=plsregress_filter2mouyi(Xtrain,Ytrain,Xtest,Ytest,LengthofFilter,ncomp)
[n,m]=size(Xtrain);
%%% initial filter
%N=LengthofFilter; % set filter size

%h=randn(1,N); %h=h/norm(h); %% 随机初始
%H=toeplitz([h zeros(1,m-N)],[1 zeros(1,m-N)]); %卷积矩阵
%%%%
for k=1:ncomp
    N= 1+randperm(LengthofFilter);  %% randomly set filter size
    N=N(1);
    h=randn(1,N); 
    [t,p,h,Xtrain,Ytrain,xloading,yloading]=plsregress_filter(Xtrain,Ytrain,LengthofFilter);
    H(k)={h};
    Ttrain(:,k)=t;    
    %train_corr(k)=corr(Ytrain,t);
    XL(k)={xloading};
    YL(k)={yloading};
    % %%%testing phase
    %for i=1:NumofComp
    t=Xtest*cell2mat(H(1,k))*cell2mat(XL(1,k));
    Ttest(:,k)=t;
    %test_corr(k)=corr(Ytest,t);
    p=(Xtest*cell2mat(H(1,k)))'*t/(t'*t);
    q=Ytest'*t/(t'*t);
    Xtestnew=Xtest*cell2mat(H(1,k))-t*p';
    Ytestnew=Ytest-t*q';
    Xtest=Xtestnew;
    Ytest=Ytestnew;
    train_err(1,k)=norm(Ytrain-Ttrain*pinv(Ttrain)*Ytrain)/sqrt(size(Xtrain,1));
    test_err(1,k)=norm(Ytest-Ttest*pinv(Ttest)*Ytest)/sqrt(size(Xtest,1));
    %%% 相关系数
    %train_corr(k)=(corr(Ytrain-Ttrain*pinv(Ttrain)*Ytrain))^2;
    %test_corr(k) = (corr(Ytest-Ttest*pinv(Ttest)*Ytest))^2;
    %%% 余弦相似度
    %train_corr(k) = 1 - pdist2(Ytrain', (Ttrain*pinv(Ttrain)*Ytrain)', 'cosine');
    %test_corr(k) = 1-pdist2(Ytest',(Ttest*pinv(Ttest)*Ytest)', 'cosine');
    test_corr(k)=cos_sim(Ytest,Ttest*pinv(Ttest)*Ytest);
    train_corr(k)=cos_sim(Ytrain, Ttrain*pinv(Ttrain)*Ytrain);
end

