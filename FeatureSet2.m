clc;
clear all;
close all;
filename = 'final_jit_shim_values.csv';
Jitt_Shimm_pd = csvread(filename,1,2);
filename2 = 'final_jit_shim_values_healthy.csv';
Jitt_Shimm_healthy = csvread(filename2,2,2);
%% For 2000 frames each
Audio_files_1 = dir('C:/Users/User1/Documents/MATLAB/Project_Phonations/Our_Dataset/Healthy_files/*.wav');
Audio_files_2 = dir('C:/Users/User1/Documents/MATLAB/Project_Phonations/Our_Dataset/PD_Files/*.wav');
    
for i=1:50
    patient_no = i;
    X = sprintf('Calculating features for audio sample of PD patient %d',patient_no);
    disp(X)
    [y{i}, Fs{i}] = audioread(Audio_files_2(i).name); % Reading the Audio1 files
    %disp(Fs{i})
    %[ym{i}, Fsm{i}] = audioread(Audio_files_2{i}); % Reading the  Audio2 files
     [x_overlap, fs] = overlap_frame(y{i},Fs{i}); 
     [~,col]= size(x_overlap);
     %disp(col)
     %%  GFCC Features -> 76 Features
     % GFCC-> Double-Delta GFCC-> 26 + 26 + 26 Features
     gfcc_sig = gfcc(x_overlap); 
    % GFCC Delta
     gfcc12_delta=zeros(26,col);
    for k=2:26
        gfcc12_delta(k,:)=gfcc_sig(k,:)-gfcc_sig(k-1,:);
    end
    % GFCC Double-Delta
    gfcc12_doubledelta = zeros(26,col);
    for k=2:26
        gfcc12_doubledelta(k,:)=gfcc12_delta(k,:)-gfcc12_delta(k-1,:);
    end
   g1 = reshape(gfcc_sig,[1,26*col]);
   g2 = reshape(gfcc12_delta,[1,26*col]);
   g3 = reshape(gfcc12_doubledelta,[1,26*col]);
   combo_gfcc = [g1 g2 g3];
    %% MFCC Features 78 Features
%      % MFCC,Delta, Double-Delta MFCC-> 26 + 26 + 26 Features
      [mfcc26 ,mfcc26_delta, mfcc26_doubledelta] = mfcc(x_overlap);
      mfcc1 = reshape(mfcc26, [1, 26*col]);
      mfcc2 = reshape(mfcc26_delta, [1, 26*col]);
      mfcc3 = reshape(mfcc26_doubledelta, [1, 26*col]);
      combo_mfcc = [mfcc1 mfcc2 mfcc3];
     combo_all{i} = [combo_gfcc combo_mfcc Jitt_Shimm_pd(i,:)];
     %disp(length(combo_all{i}))
     train_input_H_2(i,:) = combo_all{i};
end
save('train_input_PD_2.mat','train_input_PD_2')
      
      %%
function gfcc_s = gfcc(X)
[~, cols] = size(X); % Overlapped + Hamming Windowed Frames
n_overlap_frames =cols;
x_fft = zeros(512,n_overlap_frames);
gfcc13 = zeros(26,n_overlap_frames);
%disp(n_overlap_frames-1);
for k=0:n_overlap_frames-1
    %disp(k)
    % Windowing
    % FFT
    x_fft(:,k+1)= fft(X(:,k+1),512);
    % Make GT Filters
end
 
 gm2 = zeros(64,327);
 
 for e=1:cols
     x3=x_fft(:,e);
     x_h = X(:,e);
     fcfs = MakeERBFilters(44100,64,100);
      fb = ERBFilterBank(x_h', fcfs);
%       disp(size(gm))
%       disp(size(x3))
      gm2(:,e)= fb*x3;
 end
    % Find the Logarithm 
    lg = log10(gm2);
    % GFCC 13 Coefficients from DCT (2-14)
    gfcc64 = abs(dct(lg));
    gfcc13(:,:)= gfcc64(2:27,:); 
    gfcc_s = gfcc13;
end

function [mfcc_sig,mfcc_delta, mfcc_dd] = mfcc(X)
[~, cols] = size(X); % Overlapped Frames
n_overlap_frames =cols;
dft = zeros(257,n_overlap_frames);
for k=1:cols
    p = periodogram(X(:,k),[],512);
    dft(:,k) = p;    
end
Fs = 44100;
% Mel-Filter Bank %

n_filt=40;
fl=0;
fH=Fs/2;
fl_mel=0;
fH_mel = 2595*log10(1+(fH/700));
mel_points = zeros(42,1);
inc = (fH_mel-fl)/(n_filt+1);
init=0;
mel_points(1) = fl_mel;
for i=1:42
    mel_points(i)=init;
    init=init+inc;
end

hz_points=zeros(42,1);
for i=1:42
    hz_points(i)=700*(10^(mel_points(i)/2595)-1);
end

bin_hz = zeros(42,1);
for i=1:42
    bin_hz(i)= floor((512 + 1) * hz_points(i)/Fs)+1;
end

fbanks = zeros(n_filt,(floor(256 / 2 + 1)));

for i=2:41
    f_left = bin_hz(i-1);
    f_center = bin_hz(i);
    f_right = bin_hz(i+1);
    
    for j=f_left:f_center
       fbanks(i-1,j)= (j- bin_hz(i-1))/(bin_hz(i)-bin_hz(i-1));  
    end
    
    for j=f_center:f_right
       fbanks(i-1,j)= (bin_hz(i+1)-j)/(bin_hz(i+1)-bin_hz(i));  
    end
    
end
% Apply Mel-Bank onto the DFT %
Y = fbanks*dft;

% Find the Logarithm %
lg = log10(Y);

% MFCC 12 Coefficients from DCT (2-13)
mfcc1 = zeros(40,n_overlap_frames);
mfcc26 = zeros(26,n_overlap_frames);
for k=1:40
    mfcc1(k,:) = dct(lg(k,:));
   
end
mfcc26(:,:)= mfcc1(2:27,:);
% MFCC Delta
mfcc26_delta=zeros(26,n_overlap_frames);
for k=2:26
    mfcc26_delta(k-1,:)=mfcc26(k,:)-mfcc26(k-1,:);
end

% MFCC Double-Delta
mfcc26_doubledelta = zeros(26,n_overlap_frames);
for k=2:26
    mfcc26_doubledelta(k-1,:)=mfcc26_delta(k,:)-mfcc26_delta(k-1,:);
end

mfcc_sig=mfcc26;
mfcc_delta = mfcc26_delta;
mfcc_dd = mfcc26_doubledelta;
end

%%
function [overf, F_s] = overlap_frame(X,fs)
%disp(fs);
n=512; %how many samples will each frame contain
Fs = 44100;
%disp(fs)
if fs~=44100
    r = resample(X(:,1),160,441);
else
    r = X(:,1);
end
preemph = [1 -0.97];
r = filter(1,preemph,r);
r=r-mean(r); % remove DC component
s=r/max(abs(r));  %normalization
x = s;
n_overlap_frames = floor((length(x)-n)/(n/2));
x_overlap = zeros(512,n_overlap_frames);
x_hamm=zeros(512,n_overlap_frames);

for k=0:n_overlap_frames
    x_overlap(:,k+1)=x(1+(n*k/2):n*(k+1)-((k*n)/2));
    x_hamm(:,k+1)= hamming(length(x_overlap(:,k+1))).*x_overlap(:,k+1);
end
%disp(n_overlap_frames)
overf =  x_hamm(:,100:2100);
F_s = fs;
end


