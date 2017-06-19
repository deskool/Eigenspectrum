%% AUTHORS: MOHAMMAD GHASSEMI AND TUKA AL HANAI
%% ALGORITHM BASED ON PAPER 'SEIZURE PREDICTION USING EEG, SPATIOTEMPORAL CORRELATION STRUCTURE'

clear all;
number_of_channels = 6
block_length = 15
sample_rate = 256
total_data_length = 10000

s = 1 
n_d = [32,16,16,16]
delta_k =[1/64,1/16,1/4,1/1]
delta_k = round(delta_k*sample_rate)
np = 20       %number of top principal components

%% CODE STARTS HERE
Z = randn(total_data_length*sample_rate,number_of_channels);

%Generate Ztj
start_ind = 1; end_ind= 1;
j=1
while end_ind < total_data_length*sample_rate
   end_ind = start_ind + block_length*sample_rate;
   
   if end_ind < total_data_length*sample_rate
       t_start(j) = start_ind;
       t_end(j) = end_ind-1;
   else
       end_ind = total_data_length*sample_rate
       t_start(j) = start_ind;
       t_end(j) = end_ind;
   end 
   j = j+1; 
   start_ind = end_ind
end

t_start = t_start + max(delta_k)*max(n_d);
t_end = t_end + max(delta_k)*max(n_d);

%Generate Xjk
Xjk = {}
X=[]
%For each time shift
for k = 1:length(delta_k) 
    %for each block
    for j = 1:length(t_start)
        if t_end(j) < total_data_length*sample_rate
        
        %for each number of time delays per scale.
        for i = 1:n_d(k)
    
        tau = (i-1)*delta_k(k)*s;
        jk_start = t_start(j)-tau;
        jk_end = t_end(j)-tau;
        X= [X, Z(jk_start:jk_end,:)];       
        end
        Xjk{j,k} = X;
        X=[];
        end
    end
end

%Compute the C and R
ns = t_end(1) - t_start(1) + 1;
C = {};
R = {};
for j = 1:size(Xjk,1)
    
    for k = 1:size(Xjk,2)
        f = (Xjk{j,k} - repmat(mean(Xjk{j,k}),ns,1));
        g = zscore(Xjk{j,k});

        C{j,k} = 1/ns*f'*f;
        R{j,k} = 1/ns*g'*g;
          
    end
    
end

%% 
Lambda = {};
for j = 1:size(R,1)
    
    for k = 1:size(R,2)
    [~,D] = eig(R{j,k});
    eigvec = []
        for dummy = 1:length(D)
            eigvec(dummy) = D(dummy,dummy);
        end
        Lambda{j,k} = sort(eigvec,2,'descend'); 
        
    end
    
end

%% Compute pjk and hjk

rho = {};
h = {};

for j = 1:size(C,1)
    
    for k = 1:size(C,2)
        
        rho{j,k} = log(trace(C{j,k}));
        h{j,k} = log(det(C{j,k}));
        
    end
end


%% Compute the feature matrix:
features = [cell2mat(Lambda) cell2mat(rho) cell2mat(h)];
features = zscore(features);
[rlvm, frvals, frvecs, trnsfrmd, mn, dv] = pca(features)
features = trnsfrmd(:,1:np)
















